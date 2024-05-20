import os
import torch
import logging
import wandb
import numpy as np

from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from torch.nn.parallel import DistributedDataParallel as DDP


log = logging.getLogger(__name__)

def train_audio(configs, model, dataset, device='cuda'):
    # setup configs
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # set model to training mode
    model.train()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-5)

    process_bar = tqdm(range(train_configs.iterations))
    T, C = dataset.T, dataset.C
    best_psnr = 0
    best_pred = None

    # load data
    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_audio = labels.flatten().cpu().detach().numpy()
    ori_audio = (ori_audio*2) - 1 if model_configs.INPUT_OUTPUT.data_range == 1 else ori_audio # [-1,1] is the stored format

    # train
    for step in process_bar:
        preds = model(coords)
        loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        if model_configs.INPUT_OUTPUT.data_range == 2:
            preds = preds.clamp(-1, 1).view(T, C)       # clip to [-1, 1]
        else:
            preds = preds.clamp(0, 1).view(T, C)       # clip to [0, 1]
            preds = preds*2 - 1                         # [0, 1] -> [-1, 1]

        preds = preds.flatten().cpu().detach().numpy()
        psnr_score = psnr_func(preds, ori_audio, data_range=2)

        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "lr": scheduler.get_last_lr()[0]
                        }
            if step%train_configs.save_interval==0:
                log_dict["Reconstruction"] = wandb.Audio(preds, sample_rate=16000)

            wandb.log(log_dict, step=step)

        if psnr_score > best_psnr:
            best_psnr = psnr_score
            best_pred = preds
            log.info(f"Best psnr: {best_psnr:.4f}")

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, loss: {loss.item():.4f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_pred": wandb.Audio(best_pred, sample_rate=16000),
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr


def train_image(configs, model, dataset, device='cuda'):
    # setup configs
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # set model to training mode
    model.train()
    model = model.to(device)

    # opt and scheduler for iNGP
    if model_configs.name == "NGP":
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 100, gamma=0.33)
    # optimizer and scheduler
    else:
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-5)

    process_bar = tqdm(range(train_configs.iterations))
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_pred = None

    # load data
    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_img = labels.view(H, W, C).cpu().detach().numpy()
    ori_img = (ori_img + 1) / 2 if model_configs.INPUT_OUTPUT.data_range == 2 else ori_img

    # train
    for step in process_bar:
        if model_configs.name == 'ASMR' and model_configs.NET.lora:
            # ONLY APPLICABLE FOR LORA TRAINING!
            model.update_backbone(step)

        normalization_vector = torch.tensor([W, H])
        preds = model(coords, normalization_vector)
        loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        # normalize pixel values
        if model_configs.INPUT_OUTPUT.data_range == 2:
            preds = preds.clamp(-1, 1).view(H, W, C)       # clip to [-1, 1]
            preds = (preds + 1) / 2                        # [-1, 1] -> [0, 1]
        else:
            preds = preds.clamp(0, 1).view(H, W, C)       # clip to [0, 1]

        # evaluate model prediction
        preds = preds.cpu().detach().numpy()
        psnr_score = psnr_func(preds, ori_img, data_range=1)
        ssim_score = ssim_func(preds, ori_img, channel_axis=-1, data_range=1)
        
        # unsqueeze image if grayscale
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1) # Grayscale image
        
        #  log to wandb
        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0]
                        }
            # log image
            if step%train_configs.save_interval==0:
                predicted_img = Image.fromarray((preds *255).astype(np.uint8), mode=dataset_configs.color_mode)
                if predicted_img.size[0] > 512:
                    predicted_img = predicted_img.resize((512, int(512*H/W)), Image.ANTIALIAS)
                log_dict["Reconstruction"] = wandb.Image(predicted_img)

            wandb.log(log_dict, step=step)

        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = preds
            torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
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


def train_megapixel(configs, model, dataset, device='cuda'):
    # setup configs
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # set model to training mode
    model.train()
    # if cuda device id is not specified, we assume that we are training with multiple GPUs
    if device == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # opt and scheduler for iNGP
    if model_configs.name == "NGP":
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 100, gamma=0.33)
    # optimizer and scheduler
    else:
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-5)

    process_bar = tqdm(range(train_configs.iterations))
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_pred = None

    # placeholders for target and reconstructed image
    ori_img = np.zeros(dataset.get_data_shape())
    ori_img_pred = np.zeros(dataset.get_data_shape())

    # train
    for step in process_bar:
        iter_dataset = iter(dataset)
        # minibatch training over megapixel image since it doesn't fit into a single 24GB GPU
        for batch in range(len(dataset)):
            coords, labels = next(iter_dataset)
            coords, labels = coords.to(device), labels.to(device)
            preds = model(coords, [H, W])

            loss = ((preds - labels) ** 2).mean()       # MSE loss

            # backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # normalize pixel values
            if model_configs.INPUT_OUTPUT.data_range == 2:
                preds = preds.clamp(-1, 1)       # clip to [-1, 1]
                preds = (preds + 1) / 2         # [-1, 1] -> [0, 1]
            else:
                preds = preds.clamp(0, 1)       # clip to [0, 1]

            # rescale input coordinates into integers (if coordinates are normalized)
            if model_configs.INPUT_OUTPUT.coord_mode != 0:
                if model_configs.INPUT_OUTPUT.coord_mode != 1:
                    coords = (coords + 1) / 2
                coords = (coords * torch.tensor([W, H]).to(coords.device)).long()

            # copy predicted values to full image
            x0_coord = torch.round(coords[:, 0].cpu()).long().clamp(0, W-1)
            x1_coord = torch.round(coords[:, 1].cpu()).long().clamp(0, H-1)
            ori_img[x0_coord, x1_coord] = labels.detach().cpu()
            ori_img_pred[x0_coord, x1_coord] = preds.detach().cpu()

            # update loss value to wandb
            if configs.WANDB_CONFIGS.use_wandb:
                log_dict = {"loss": loss.item()}
                wandb.log(log_dict, step=step)
        
        # unsqueeze image if grayscale
        if ori_img_pred.shape[-1] == 1:
            ori_img_pred = ori_img_pred.squeeze(-1) # Grayscale image

        # update learning rate
        scheduler.step()

        # log to wandb
        if configs.WANDB_CONFIGS.use_wandb and step%train_configs.save_interval==0:
            psnr_score = psnr_func(ori_img_pred, ori_img, data_range=1)
            ssim_score = ssim_func(ori_img_pred, ori_img, channel_axis=-1, data_range=1)
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0]
                        }
            # convert model prediction to Image
            predicted_img = Image.fromarray((ori_img_pred *255).astype(np.uint8), mode=dataset_configs.color_mode)
            # resize image if too big
            if predicted_img.size[0] > 512:
                predicted_img_resized = predicted_img.resize((512, 512), Image.ANTIALIAS)
            
            # log image to wandb
            log_dict["Reconstruction"] = wandb.Image(predicted_img_resized)

            wandb.log(log_dict, step=step)

        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = predicted_img
            best_pred.save(os.path.join('outputs', out_dir, 'best_pred.png'), format='png')

        # save full res image for manuscript
        if step in [99, 499]:
            predicted_img.save(os.path.join('outputs', out_dir, f'pred_{step}.png'), format='png')

        # update progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr, best_ssim


def train_video(configs, model, dataset, device='cuda'):
    # setup configs
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # set model to training mode
    model.train()
    model = model.to(device)
    
    # opt and scheduler for iNGP
    if model_configs.name == "NGP":
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr, betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 100, gamma=0.33)
    # optimizer and scheduler
    else:
        opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-5)

    process_bar = tqdm(range(train_configs.iterations), disable=True)
    T, H, W, C = dataset.T, dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    psnr_std, ssim_std = 0, 0

    # train
    for step in process_bar:
        iter_dataset = iter(dataset)
        for batch in range(len(dataset)):
            coords_batch, labels_batch = next(iter_dataset)
            coords_batch, labels_batch = coords_batch.to(device), labels_batch.to(device)
            if model_configs.name == "SIREN" or model_configs.name == "LOE":
                preds_batch = model(coords_batch, labels=[T, H, W])
            else:
                preds_batch = model(coords_batch, labels_batch)

            loss = ((preds_batch - labels_batch) ** 2).mean()       # MSE loss

            # backprop
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)     # empirically necessary in our test runs
            opt.step()
            
            if configs.WANDB_CONFIGS.use_wandb:
                log_dict = {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}
                wandb.log(log_dict, step=step)

            # udpate progress bar
            process_bar.set_description(f"iter: {step}:{batch} | loss: {loss.item():.4f}")

            del loss
            del preds_batch

        scheduler.step()

        if configs.WANDB_CONFIGS.use_wandb and step%train_configs.val_interval==0:

            # Evaluate and visualize on the entire video
            model.eval()
            with torch.no_grad():
                ori_video = np.zeros(dataset.get_data_shape())
                val_ori_video_pred = np.zeros(dataset.get_data_shape())
                # train
                coords_ls = []
                labels_ls = []
                iter_dataset = iter(dataset)
                for batch in range(len(dataset)):
                    coords_batch, labels_batch = next(iter_dataset)
                    if batch == 0 or batch % 2 == 0:
                        coords = coords_batch
                        labels = labels_batch
                    else:
                        coords = torch.cat((coords, coords_batch), dim=0)
                        labels = torch.cat((labels, labels_batch), dim=0)

                    if batch != 0 and (batch+1) % 2 == 0:
                        coords_ls.append(coords)
                        labels_ls.append(labels)
                
                for big_batch in range(len(coords_ls)):
                    coords_batch = coords_ls[big_batch]
                    labels_batch = labels_ls[big_batch]
                    coords_batch, labels_batch = coords_batch.to(device), labels_batch.to(device)
                    
                    if model_configs.name == "SIREN":
                        preds_batch = model(coords_batch, labels=[T, H, W])
                    else:
                        preds_batch = model(coords_batch, labels_batch)

                    if model_configs.INPUT_OUTPUT.data_range == 2:
                        preds_batch = preds_batch.clamp(-1, 1)
                        preds_batch = (preds_batch + 1) / 2                        # [-1, 1] -> [0, 1]
                        labels_batch = labels_batch.clamp(-1, 1)
                        labels_batch = (labels_batch + 1)/2
                    else:
                        preds_batch = preds_batch.clamp(0, 1)                      # clip to [0, 1]
                        labels_batch = labels_batch.clamp(0, 1)

                    x0_coord = coords_batch[:, 0].cpu().int()
                    x1_coord = coords_batch[:, 1].cpu().int()
                    x2_coord = coords_batch[:, 2].cpu().int()
                    ori_video[x0_coord, x1_coord, x2_coord] = labels_batch.detach().cpu()
                    val_ori_video_pred[x0_coord, x1_coord, x2_coord] = preds_batch.detach().cpu()

            print("Calculating per frame psnr...")
            psnr_frames = []
            ssim_frames = []
            for frame in range(T):
                ori_video_frame = ori_video[frame]
                val_ori_video_pred_frame = val_ori_video_pred[frame]
                val_psnr = psnr_func(val_ori_video_pred_frame, ori_video_frame, data_range=1)
                val_ssim = ssim_func(val_ori_video_pred_frame, ori_video_frame, channel_axis=-1, data_range=1)
                psnr_frames.append(val_psnr)
                ssim_frames.append(val_ssim) 

            val_psnr, val_ssim = np.mean(psnr_frames), np.mean(ssim_frames)
            val_psnr_std, val_ssim_std = np.std(psnr_frames), np.std(ssim_frames)
            
            ori_video = np.moveaxis(ori_video, 3, 1)
            val_ori_video_pred = np.moveaxis(val_ori_video_pred, 3, 1)
            
            if val_psnr > best_psnr:
                best_psnr, best_ssim = val_psnr, val_ssim
                psnr_std, ssim_std = val_psnr_std, val_ssim_std

            if configs.WANDB_CONFIGS.use_wandb:
                wandb.log(
                        {
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_psnr_std": psnr_std,
                        "best_ssim_std": ssim_std,
                        "pred_video": wandb.Video(val_ori_video_pred*255, fps=4, format='gif'),
                        "gt_video": wandb.Video(ori_video*255, fps=4, format='gif')
                        }, 
                    step=step)
            log.info(f"Best psnr: {best_psnr:.4f}+={psnr_std:.4f}, ssim: {best_ssim*100:.4f}+={ssim_std*100:.4f}")

        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "best_psnr_std": psnr_std,
                "best_ssim_std": ssim_std
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}+={psnr_std:.4f}, ssim: {best_ssim*100:.4f}+={ssim_std*100:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr, best_ssim