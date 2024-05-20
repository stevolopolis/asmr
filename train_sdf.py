import os
import yaml
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from utils import seed_everything, get_dataset, get_model
from utils_sdf import create_pointcloud, create_mesh
import wandb

log = logging.getLogger(__name__)

def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs

def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))


def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # opt and scheduler for BACON training
    if model_configs.name == 'BACON_old':
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10000, gamma=0.1)
    elif model_configs.name == 'LOE_old':
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr, betas=(0.9, 0.995))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 5000, gamma=0.1)
    # optimizer and scheduler
    else:
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prepare training settings
    model.train()
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    best_loss, best_sdf, best_inter, best_normal, best_grad = 0, 0, 0, 0, 0
    best_pred = None

    # train
    losses = []
    for step, (x, y) in enumerate(dataset):
        # Enable us to compute gradients w.r.t. coordinates (to compare with surface normal)
        x = x.to(device)
        x_org = x.clone().detach().requires_grad_(True)
        # load data, pass through model, get loss
        preds = model(x_org)

        gradient = compute_gradient(preds, x_org)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        zeros = torch.zeros_like(preds).to(device)
        sdf_constraint = torch.abs(torch.where(y['sdf'].to(device) != -1, preds, zeros)).mean() * 3e3
        inter_constraint = torch.where(y['sdf'].to(device) != -1, zeros, torch.exp(-1e2 * torch.abs(preds))).mean() * 1e2
        normal_constraint = torch.where(y['sdf'].to(device) != -1, 1 - F.cosine_similarity(gradient, y['normals'].to(device), dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]).to(device)).mean() * 1e2
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1).mean() * 5e1

        loss = sdf_constraint + inter_constraint + normal_constraint + grad_constraint
                
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {"loss": loss.item(),
                        "sdf": sdf_constraint,
                        "inter": inter_constraint,
                        "normal": normal_constraint,
                        "grad": grad_constraint}
            if step % train_configs.save_interval == 0:
                if step == 0:
                    pc = dataset.get_pointcloud()
                else:
                    #pc = create_pointcloud(model, dataset_configs.vox_res, dataset_configs.batch_size)
                    create_mesh(model, "statuette_siren", dataset_configs.vox_res, dataset_configs.batch_size)
                #log_dict["Reconstruction"] = wandb.Object3D(pc)

            wandb.log(log_dict, step=step)

        if loss.item() > best_loss:
            best_loss, best_sdf, best_inter, best_normal, best_grad = loss.item(), sdf_constraint, inter_constraint, normal_constraint, grad_constraint
            best_pred = preds

        # update progress bar
        process_bar.set_description(f"loss: {loss.item():.2f}, sdf: {sdf_constraint:.2f}, inter: {inter_constraint:.2f}, normal: {normal_constraint:.2f}, grad: {grad_constraint:.2f}")
        process_bar.update()
        # update scheduler
        scheduler.step()
        
    print("Training finished!")
    print(f"Best loss: {best_loss:.2f}, sdf: {best_sdf:.2f}, inter: {best_inter:.2f}, normal: {best_normal:.2f}, grad: {best_grad:.2f}")
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.log(
                {
                "best_loss": best_loss,
                "best_sdf": best_sdf,
                "best_inter": best_inter,
                "best_normal": best_normal,
                "best_grad": best_grad,
                "best_pred": wandb.Image(Image.fromarray((best_pred*255).astype(np.uint8), mode=dataset_configs.color_mode)),
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best loss: {best_loss:.2f}, sdf: {best_sdf:.2f}, inter: {best_inter:.2f}, normal: {best_normal:.2f}, grad: {best_grad:.2f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

    return best_loss, best_sdf, best_inter, best_normal, best_grad


@hydra.main(version_base=None, config_path='config', config_name='train_sdf')
def main(configs):
    seed_everything(42)

    configs = EasyDict(configs)
    # data = configs.DATASET_CONFIGS.file_path.split('/')[-1].split('.')[0]
    # configs.TRAIN_CONFIGS.out_dir = f"{configs.model_config.name}_{data}_proxy"
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    # wandb
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.init(
            project=configs.WANDB_CONFIGS.wandb_project,
            entity=configs.WANDB_CONFIGS.wandb_entity,
            config=configs,
            group=configs.WANDB_CONFIGS.group,
            name=configs.TRAIN_CONFIGS.out_dir,
        )

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")

    # train
    loss, sdf, inter, normal, grad = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)

    return loss, sdf, inter, normal, grad, n_params

if __name__=='__main__':
    main()