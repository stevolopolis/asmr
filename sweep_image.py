import os
import yaml
import shutil
import torch
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from utils import seed_everything, get_dataset, get_model
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
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


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # opt and scheduler for BACON training
    if model_configs.name == 'BACON':
        opt = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10000, gamma=0.1)
    # optimizer and scheduler
    else:
        opt = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prepare training settings 
    model.train()
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    H, W = dataset.H, dataset.W
    C = dataset.dim_out
    best_psnr, best_ssim = 0, 0
    best_pred = None

    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_img = labels.view(H, W, C).cpu().detach().numpy()
    ori_img = (ori_img + 1) / 2 if model_configs.INPUT_OUTPUT.data_range == 2 else ori_img

    # train
    for step in process_bar:
        preds = model(coords, labels)
        # For BACON training
        if type(preds) == list:
            loss = [(out - labels)**2 for out in preds]
            loss = torch.stack(loss).mean()
            preds = preds[-1]
        else:
            loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        if model_configs.INPUT_OUTPUT.data_range == 2:
            preds = preds.clamp(-1, 1).view(H, W, C)       # clip to [-1, 1]
            preds = (preds + 1) / 2                        # [-1, 1] -> [0, 1]
        else:
            preds = preds.clamp(0, 1).view(H, W, C)       # clip to [0, 1]

        preds = preds.cpu().detach().numpy()
        psnr_score = psnr_func(preds, ori_img, data_range=1)
        ssim_score = ssim_func(preds, ori_img, channel_axis=-1, data_range=1)

        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0]
                        }
            if step%train_configs.save_interval==0:
                predicted_img = Image.fromarray((preds*255).astype(np.uint8), mode=dataset_configs.color_mode)
                log_dict["Reconstruction"] = wandb.Image(predicted_img)

            wandb.log(log_dict, step=step)

        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = preds

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    wandb.log(
            {
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_pred": wandb.Image(Image.fromarray((best_pred*255).astype(np.uint8), mode=dataset_configs.color_mode)),
            }, 
        step=step)
    wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr, best_ssim


@hydra.main(version_base=None, config_path='config', config_name='train_image')
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
    # change model configuration based on sweep config
    configs.model_config.NET.num_freq = wandb.config.num_freq

    configs.TRAIN_CONFIGS.iterations = 2000

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")

    # train
    psnr, ssim = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)

    return psnr, ssim, n_params

if __name__=='__main__':
    sweep_config = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'psnr'},
        'parameters':
        {
            'lr': {'values': [1e-2, 1e-3, 1e-4]},
            'num_freq': {'values': [8, 16, 32]}
        }
    }
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project='loe_image_sweep',
        entity='taco-wacv'
    )

    wandb.agent(sweep_id, function=main, count=30)