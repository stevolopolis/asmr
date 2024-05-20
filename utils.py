import random 
import yaml
import shutil
import numpy as np
import torch
from dataset import *
import torchvision
from omegaconf import OmegaConf


def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs

def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))


def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed) # for random partitioning
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def visualize_activations(activation, H=512, W=768, name='temp_activations', quality=60, downsample=5):
    # make torchvision grid and save as image
    activation = (activation - activation.min(dim=0)[0])/(activation.max(dim=0)[0] - activation.min(dim=0)[0])
    activation = rearrange(activation, '(h w) c -> c 1 h w', h=H, w=W)
    activation = activation.detach().cpu()
    grid = torchvision.utils.make_grid(activation, nrow=16, padding=0)
    img = torchvision.transforms.ToPILImage()(grid)
    img = img.resize((img.size[0]//downsample, img.size[1]//downsample))    # reduce size to save space
    img.save(f"{name}.jpg", format='JPEG', quality=quality)    


def ds_profiling(model, data_shape, device=0):
    from deepspeed.accelerator import get_accelerator
    from deepspeed.profiling.flops_profiler import get_model_profile
    # computational cost in number of multiply-accumulates (MACs) per sample
    spatial_size = data_shape[:-1]
    with get_accelerator().device(device):
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=data_shape, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=3, # the number of warm-ups before measuring the time of each module
                                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
    return flops/np.prod(spatial_size), macs/np.prod(spatial_size), params


def get_dataset(dataset_configs, input_output_configs):
    if  dataset_configs.data_type == "audio":
        dataset = AudioFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "video":
        dataset = VideoFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "image":
        dataset = ImageFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "cameraman":
        dataset = CameraFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "sdf":
        dataset = PointCloud(dataset_configs)
    elif dataset_configs.data_type == "megapixel":
        dataset = BigImageFile(dataset_configs, input_output_configs)
    
    return dataset


def get_model(model_configs, dataset):
    if model_configs.name == 'ASMR':
        from models.asmr import ASMR
        model = ASMR(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            data_size = dataset.get_data_shape()[:-1],
            asmr_configs=model_configs
        )
    elif model_configs.name == 'SIREN':
        from models.siren import Siren
        model = Siren(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            siren_configs=model_configs
        )
    elif model_configs.name == 'FFN':
        from models.ffn import FFN
        model = FFN(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            ffn_configs=model_configs
        )
    elif model_configs.name == "WIRE":
        from models.wire import Wire
        model = Wire(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           wire_configs=model_configs
        )
    elif model_configs.name == 'KILONERF':
        from models.kilonerf import KiloNerf
        model = KiloNerf(
            in_features=dataset.dim_in,
            out_features=dataset.dim_out,
            data_size = dataset.get_data_shape()[:-1],
            kilo_configs=model_configs
        )
    elif model_configs.name == 'NGP':
        from models.ngp import NGP
        model = NGP(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            data_size=dataset.get_data_shape()[:-1],
            ngp_configs=model_configs
        )
    elif model_configs.name == "LOE":
        from models.loe import LoE
        model_configs.NET.img_dim = (dataset.H, dataset.W)
        preset_dim = model_configs.NET.dimensions[-1]
        if dataset.H > dataset.W:
            model_configs.NET.dimensions[-1] = [max(preset_dim), min(preset_dim)] if dataset.dim_in == 2 else [1, 3, 2]
        elif dataset.H < dataset.W:
            model_configs.NET.dimensions[-1] = [min(preset_dim), max(preset_dim)] if dataset.dim_in == 2 else [1, 2, 3]
        print(model_configs.NET.dimensions)

        model = LoE(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           loe_configs=model_configs
        )
    elif model_configs.name == "LOE_SIREN":
        from models.loe import LoE_SIREN
        model_configs.NET.img_dim = (dataset.H, dataset.W)
        preset_dim = model_configs.NET.dimensions[-1]
        if dataset.H > dataset.W:
            model_configs.NET.dimensions[-1] = [max(preset_dim), min(preset_dim)] if dataset.dim_in == 2 else [1, 3, 2]
        elif dataset.H < dataset.W:
            model_configs.NET.dimensions[-1] = [min(preset_dim), max(preset_dim)] if dataset.dim_in == 2 else [1, 2, 3]
        #else:
        #    model_configs.NET.dimensions[-1] = [2, 2] if dataset.dim_in == 2 else [1, 2, 2]
        print(model_configs.NET.dimensions)

        model = LoE_SIREN(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           loe_configs=model_configs
        )
    else:
        raise NotImplementedError(f"Model {model_configs.name} not implemented")
            
    return model