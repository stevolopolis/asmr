import hydra
import wandb
from easydict import EasyDict

from utils import seed_everything, save_src_for_reproduce, get_dataset, get_model

config_name = None

@hydra.main(version_base=None, config_path='config', config_name=config_name)
def main(configs):
    print("==============================")
    if configs.DATASET_CONFIGS.data_type == "audio":
        from train_utils import train_audio as train
        print("Training AUDIO")
        print("=====================================")
    elif configs.DATASET_CONFIGS.data_type in ["image", "cameraman"]:
        from train_utils import train_image as train
        print("Training IMAGE")
        print("=====================================")
    elif configs.DATASET_CONFIGS.data_type == "megapixel":
        # from train_utils import train_megapixel as train
        from train_utils import train_megapixel as train
        print("Training MEGAPIXEL")
    elif configs.DATASET_CONFIGS.data_type == "video":
        # from train_utils import train_megapixel as train
        from train_utils import train_video as train
        print("Training VIDEO")
    else:
        raise NotImplementedError(f"Data type {configs.DATASET_CONFIGS.data_type} is not implemented")    
    print("==============================")

    # update modality-specific model config
    if configs.TRAIN_CONFIGS.model_config_type == "AUDIO" and hasattr(configs.model_config, "INPUT_OUTPUT_AUDIO") and hasattr(configs.model_config, "NET_AUDIO"):
        configs.model_config.INPUT_OUTPUT = configs.model_config.INPUT_OUTPUT_AUDIO
        configs.model_config.NET = configs.model_config.NET_AUDIO
    elif configs.TRAIN_CONFIGS.model_config_type == "IMG" and hasattr(configs.model_config, "INPUT_OUTPUT_IMG") and hasattr(configs.model_config, "NET_IMG"):
        configs.model_config.INPUT_OUTPUT = configs.model_config.INPUT_OUTPUT_IMG
        configs.model_config.NET = configs.model_config.NET_IMG
    elif configs.TRAIN_CONFIGS.model_config_type == "MEGA" and hasattr(configs.model_config, "INPUT_OUTPUT_MEGA") and hasattr(configs.model_config, "NET_MEGA"):
        configs.model_config.INPUT_OUTPUT = configs.model_config.INPUT_OUTPUT_MEGA
        configs.model_config.NET = configs.model_config.NET_MEGA
    elif configs.TRAIN_CONFIGS.model_config_type == "VIDEO" and hasattr(configs.model_config, "INPUT_OUTPUT_VIDEO") and hasattr(configs.model_config, "NET_VIDEO"):
        configs.model_config.INPUT_OUTPUT = configs.model_config.INPUT_OUTPUT_VIDEO
        configs.model_config.NET = configs.model_config.NET_VIDEO
    else:
        print("=====================================")
        print("This model does not have an specific config for modality: %s." % str.upper(configs.DATASET_CONFIGS.data_type))
        print("=====================================")

    # update config from bash script
    if hasattr(configs.TRAIN_CONFIGS, "num_layers"):
        configs.model_config.NET.num_layers = configs.TRAIN_CONFIGS.num_layers
        print("Updated num_layers: ", configs.TRAIN_CONFIGS.num_layers)
    if hasattr(configs.TRAIN_CONFIGS, "dimensions"):
        configs.model_config.NET.dimensions = configs.TRAIN_CONFIGS.dimensions
        print("Updated dimensions: ", configs.TRAIN_CONFIGS.dimensions)
    if hasattr(configs.TRAIN_CONFIGS, "dim_hidden"):
        configs.model_config.NET.dim_hidden = configs.TRAIN_CONFIGS.dim_hidden
        print("Updated dim_hidden: ", configs.TRAIN_CONFIGS.dim_hidden)    

    # load config file
    configs = EasyDict(configs)
    # set random seed
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # save src and config for reproduction
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)


    if hasattr(configs.TRAIN_CONFIGS, "tile_dims"):
        configs.model_config.NET.dimensions = configs.TRAIN_CONFIGS.tile_dims
    if hasattr(configs.model_config.NET, "lr"):
        configs.TRAIN_CONFIGS.lr = configs.model_config.NET.lr

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)

    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")
    
    # wandb
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.init(
            project=configs.WANDB_CONFIGS.wandb_project,
            entity=configs.WANDB_CONFIGS.wandb_entity,
            config=configs,
            group=configs.WANDB_CONFIGS.group,
            name=configs.TRAIN_CONFIGS.out_dir,
        )

        wandb.run.summary['n_params'] = n_params

    # train
    train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)


if __name__=='__main__':
    main()