import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchaudio
from PIL import Image
from einops import rearrange
import numpy as np
import skimage
import skimage


def uniform_grid_sampler(data_size: torch.tensor, grid_size: torch.tensor, samples_per_iteration: int):
        """
        Given a uniformly sampled set of points (samples), partition it into equal subsets with the same number
        of points in each grid.

        n_iterations = n_samples // samples_per_iteration

        Input:
            - samples: meshgrid of positive integer coordinate values
            - data_size: (d, d, ..., d) dimensions of the data. For now, assume each axis to share same dim
        
        For example:
            - Given samples (i.e. coordinates) from a 2D image with size (512**2, 2)
            - Given, grid_size = (8, 8) and samples_per_iteration = 2**15 = 32768
            - Return a tensor <partitioned_samples> of size (n_iterations, 32768, 2)
                where for each i, partitioned_samples[i] contains the same number
                of points in each of the 64 grids.
        """
        n_data = torch.prod(data_size)
        n_grids = torch.prod(grid_size).item()
        
        assert n_data % samples_per_iteration == 0, "samples_per_iteration must be a factor of n_data. data_size: %s n_data: %s\t samples: %s" % (data_size, n_data, samples_per_iteration)
        assert samples_per_iteration % n_grids == 0, "samples_per_iteration must be a factor of n_grids"

        data_dim = len(data_size)
        n_iterations = n_data // samples_per_iteration
        samples_per_grid = n_data // n_grids
        grid_dim = (data_size / grid_size).int()
        samples_per_grid_per_iter = samples_per_iteration // n_grids

        subsamples = torch.stack(torch.meshgrid([torch.linspace(0, grid_dim[i]-1, grid_dim[i]) for i in range(data_dim)], indexing='ij'), dim=-1).view(-1, data_dim)
        # Randomize the subsamples
        random_idx = torch.randperm(len(subsamples))
        subsamples = subsamples[random_idx]
        # Partition the subsamples into n_iterations
        subsamples = subsamples.view(n_iterations, -1, data_dim)
        # Generate meshgrid of grid coordinates
        grid_coords = torch.stack(torch.meshgrid([torch.linspace(0, grid_size[i]-1, grid_size[i]) for i in range(data_dim)], indexing='ij'), dim=-1).view(-1, data_dim)
        # Get n_grids copies of the meshgrid of coordinates (these coordinates are bounded by the grid size)
        subsamples = subsamples.unsqueeze(1).repeat(1, n_grids, 1, 1)
        # Padding
        grid_paddings = grid_coords * grid_dim
        # Multiple the meshgrid of coordinates by the grid coordinates to get the actual coordinates (bounded by data_dim)
        subsamples = subsamples + grid_paddings.unsqueeze(0).unsqueeze(2) 
        # Target shape: (n_iterations, n_grids * 512, data_dim)
        partitioned_samples = subsamples.view(n_iterations, -1, data_dim)
        
        return partitioned_samples

class AudioFileDataset(torchaudio.datasets.LIBRISPEECH):
    """LIBRISPEECH dataset without labels.

    Args:
        patch_shape (int): Shape of patch to use. If -1, uses all data (no patching).
        num_secs (float): Number of seconds of audio to use. If -1, uses all available
            audio.
        normalize (bool): Whether to normalize data to lie in [0, 1].
    """

    def __init__(
        self,
        dataset_configs,
        input_output_configs
    ):
        super().__init__(root='/home/jason/dev/taco-wacv/data', url='test-clean', download=False)

        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range

        # LibriSpeech contains audio 16kHz rate
        self.sample_rate = 16000
        self.num_secs = dataset_configs.num_secs
        self.num_waveform_samples = int(self.num_secs * self.sample_rate)
        self.sample_idx = dataset_configs.sample_idx

        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        self.labels = super().__getitem__(self.sample_idx)[0].float()

        # Normalize data to lie in [0, 1]
        if self.data_range == 1:
            self.labels = (self.labels + 1) / 2

        # Extract only first num_waveform_samples from waveform
        if self.num_secs != -1:
            # Shape (channels, num_waveform_samples)
            self.labels = self.labels[:, : self.num_waveform_samples].view(-1, 1)

        self.T, self.C = self.num_waveform_samples, 1
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, self.T-1, self.T)] # [0, T-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., self.T)] # [0, 1]
        elif self.coord_mode == 2:
            grid = [torch.linspace(-5., 5., self.T)] # [-5, 5] following coin++
        elif self.coord_mode == 3:
            grid = [torch.linspace(-1., 1. - 1e-6, self.T)] # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, self.T)] # [0.5, 0.5]
        else:
            raise NotImplementedError

        self.coords = torch.stack(
            torch.meshgrid(grid),
            dim=-1,
        ).view(-1, 1)

        self.dim_in = 1
        self.dim_out = 1
        

    def __len__(self):
        return 1

    def get_data_shape(self):
        return (self.T, self.C)

    def get_data(self):
        return self.coords, self.labels


class ImageFileDataset(Dataset):
    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range
        self.color_mode = dataset_configs.color_mode
        
        if 'camera' in dataset_configs.file_path:
            img = Image.fromarray(skimage.data.camera())
            assert dataset_configs.color_mode == 'L', "camera dataset is in grayscale"
        else:
            img = Image.open(dataset_configs.file_path)
            img = img.convert(self.color_mode)
        
        if dataset_configs.img_size is not None:
            img = img.resize(dataset_configs.img_size)

        self.img = img
        self.img_size = img.size
        print("Image size: ", self.img_size)

        img_tensor = ToTensor()(img) # [0, 1]

        if self.data_range == 2:
            img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        img_tensor = rearrange(img_tensor, 'c h w -> (h w) c')
        self.labels = img_tensor

        W, H = self.img_size
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, H-1, H), torch.linspace(0.0, W-1, W)] # [0, H-1] x [0, W-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., H), torch.linspace(0., 1., W)] # [0, 1]^2
        elif self.coord_mode == 2:
            grid = [torch.linspace(-1., 1., H), torch.linspace(-1., 1., W)] # [-1, 1]^2
        elif self.coord_mode == 3:
            grid = [torch.linspace(-1., 1. - 1e-6, H), torch.linspace(-1., 1. - 1e-6, W)] # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W)] # [0.5, 0.5]^2
        else:
            raise ValueError("Invalid coord_mode")

        self.coords = torch.stack(
            torch.meshgrid(grid, indexing='ij'),
            dim=-1,
        ).view(-1, 2)

        self.H, self.W = H, W
        self.dim_in = 2
        self.dim_out = 3 if self.color_mode == 'RGB' else 1
        self.C = self.dim_out

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_h(self):
        return self.H
    
    def get_w(self):
        return self.W
    
    def get_c(self):
        return self.C

    def get_data_shape(self):
        return (self.H, self.W, self.C)

    def get_data(self):
        return self.coords, self.labels


# Cameraman unit test for LoE
class CameraFileDataset(Dataset):
    """
    Sklearn Camera Man Image.
    
    Mainly for sanity check.
    """
    def __init__(self, dataset_configs, input_output_configs):
        import skimage
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range
        self.color_mode = dataset_configs.color_mode
        
        img = Image.fromarray(skimage.data.camera())
        img = img.convert(self.color_mode)
        self.img = img
        self.img_size = img.size
        img_tensor = ToTensor()(img) # [0, 1]

        if self.data_range == 2:
            img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        img_tensor = rearrange(img_tensor, 'c h w -> (h w) c')
        self.labels = img_tensor

        W, H = self.img_size
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, H-1, H), torch.linspace(0.0, W-1, W)] # [0, H-1] x [0, W-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., H), torch.linspace(0., 1., W)] # [0, 1]^2
        elif self.coord_mode == 2:
            grid = [torch.linspace(-1., 1., H), torch.linspace(-1., 1., W)] # [-1, 1]^2
        elif self.coord_mode == 3:
            grid = [torch.linspace(-1., 1. - 1e-6, H), torch.linspace(-1., 1. - 1e-6, W)] # [0, 0.999999]^2
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W)] # [0.5, 0.5]^2
        else:
            raise ValueError("Invalid coord_mode")

        self.coords = torch.stack(
            torch.meshgrid(grid, indexing='ij'),
            dim=-1,
        ).view(-1, 2)

        self.H, self.W = H, W
        self.dim_in = 2
        self.dim_out = 1
        self.C = self.dim_out

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_img_h(self):
        return self.H
    
    def get_img_w(self):
        return self.W
    
    def get_img_c(self):
        return self.C

    def get_data_shape(self):
        return (self.H, self.W, self.C)

    def get_data(self):
        return self.coords, self.labels


class PointCloud(torch.utils.data.Dataset):
    def __init__(self, dataset_configs):
        super().__init__()

        print("Loading point cloud")
        self.point_cloud = np.genfromtxt(dataset_configs.file_path)
        print("Finished loading point cloud")
        print(self.point_cloud.shape)

        self.dim_in = 3
        self.dim_out = 1
        self.H = 1
        self.W = 1
        self.Z = 1

        coords = self.point_cloud[:, :3]
        self.normals = self.point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if dataset_configs.keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = dataset_configs.on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return torch.from_numpy(coords).float(), {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

    def get_data_shape(self):
        return self.coords.shape

    def get_pointcloud(self):
        return self.point_cloud[:, :3]
    

class BigImageFile(Dataset):
    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 1000000000
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range
        self.color_mode = dataset_configs.color_mode
        self.grid_dims = input_output_configs.grid_dims

        self.gpu_max_coords = dataset_configs.max_coords
        self.img = Image.open(dataset_configs.file_path)
        if hasattr(dataset_configs, 'img_size') and dataset_configs.img_size is not None:
            self.img = self.img.resize(dataset_configs.img_size)

        self.img = self.img.convert(self.color_mode)
        self.C = len(self.img.mode)

        self.W, self.H = self.img.size
        
        self.img = torch.tensor(np.array(self.img))
        self.img = self.img / 255      #(self.img / 255 - 0.5) * 2

        print("Image size: ", self.img.shape)

        self.coords = uniform_grid_sampler(torch.tensor([self.H, self.W]), torch.tensor(self.grid_dims), self.gpu_max_coords)

        if self.coord_mode == 0:
            pass
        elif self.coord_mode == 1:
            self.coords = self.coords / torch.tensor([self.H, self.W])      # [0, 1]^3
        elif self.coord_mode == 2:
            self.coords = (self.coords / torch.tensor([self.H, self.W]) - 0.5) * 2       # [-1, 1]^3
        elif self.coord_mode == 3:
            self.coords = (self.coords / torch.tensor([self.H, self.W]) - 0.5) * 2       # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            self.coords = self.coords / torch.tensor([self.H, self.W]) - 0.5    # [0.5, 0.5]^2
        else:
            raise ValueError("Invalid coord_mode")

        self.dim_in = 2
        self.dim_out = 3 if self.color_mode == 'RGB' else 1

        self.sampled_coords = [self.coords[i] for i in range(self.coords.shape[0])]
        self.sampled_img = [self.img[self.coords[i].long()[:, 0], self.coords[i].long()[:, 1], :].view(-1, self.C) for i in range(self.coords.shape[0])]

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.sampled_coords[idx], self.sampled_img[idx]
    
    def get_h(self):
        return self.H
    
    def get_w(self):
        return self.W
    
    def get_c(self):
        return self.C

    def get_data_shape(self):
        return (self.H, self.W, self.C)
    

class VideoFileDataset(Dataset):
    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range
        self.grid_dims = input_output_configs.grid_dims

        self.gpu_max_coords = dataset_configs.batch_size

        if dataset_configs.file_path.split('.')[-1] == 'npy': # file_path is .npy array
            video_tensor = torch.tensor(np.load(dataset_configs.file_path)) # video_file as numpy array [T, H, W, C]
        elif os.path.isdir(dataset_configs.file_path): # file_path is directory containing .png frames
            if dataset_configs.file_path.split('/')[-1] == "shakendry":
                video_tensor = self.png2tensor(dataset_configs.file_path, 30, True, max_frames=75)
            else:
                video_tensor = self.png2tensor(dataset_configs.file_path, 30, True, max_frames=150)
            video_tensor = video_tensor / 255   # Normalize to [0, 1]
        else:
            raise ValueError("File path is not accepted.")

        self.T, self.H, self.W, self.C = video_tensor.shape
        self.video = video_tensor

        self.coords = uniform_grid_sampler(torch.tensor([self.T, self.H, self.W]), torch.tensor(self.grid_dims), self.gpu_max_coords)
        
        if self.coord_mode == 0:
            pass
        elif self.coord_mode == 1:
            self.coords = self.coords / torch.tensor([self.T, self.H, self.W])      # [0, 1]^3
        elif self.coord_mode == 2:
            self.coords = (self.coords / torch.tensor([self.T, self.H, self.W]) - 0.5) * 2       # [-1, 1]^3
        elif self.coord_mode == 3:
            self.coords = (self.coords / torch.tensor([self.T, self.H, self.W]) - 0.5) * 2       # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            self.coords = self.coords / torch.tensor([self.T, self.H, self.W]) - 0.5    # [0.5, 0.5]^2
        else:
            raise ValueError("Invalid coord_mode")

        self.labels = self.video.view(-1, self.C)

        self.dim_in = 3
        self.dim_out = 3
    
        self.sampled_coords = [self.coords[i] for i in range(self.coords.shape[0])]
        self.sampled_video = [self.video[self.coords[i].long()[:, 0],
                                         self.coords[i].long()[:, 1],
                                         self.coords[i].long()[:, 2],
                                         :
                                         ].view(-1, self.C) for i in range(self.coords.shape[0])]

    def png2tensor(self, dir_path, fps, downscale=False, max_frames=150):
        ori_fps = 120
        interval = ori_fps // fps
        n_frames = len(os.listdir(dir_path))
        for frame in range(1, n_frames+1):
            if frame % interval == 1:
                frame_im = Image.open(os.path.join(dir_path, "f%s.png" % str(frame).zfill(5)))
                if downscale: # Downscale by factor of 4
                    frame_im = frame_im.resize((480, 270), Image.LANCZOS)
                frame_arr = np.array(frame_im)
                if frame == 1:
                    vid_arr = np.expand_dims(frame_arr, 0)
                else:
                    frame_arr = np.expand_dims(frame_arr, 0)
                    vid_arr = np.concatenate((vid_arr, frame_arr), axis=0)

        vid_arr = vid_arr[:max_frames]  # Set to 75 for shakendry
        vid_tensor = torch.tensor(vid_arr)
        print("Extracted from %d frames" % n_frames)
        print("Video tensor size:", vid_tensor.shape)
        return vid_tensor

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.sampled_coords[idx], self.sampled_video[idx]

    def get_t(self):
        return self.T

    def get_h(self):
        return self.H
    
    def get_w(self):
        return self.W
    
    def get_c(self):
        return self.C

    def get_data_shape(self):
        return (self.T, self.H, self.W, self.C)

    def get_data(self):
        return self.coords, self.labels
