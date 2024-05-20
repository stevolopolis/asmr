import torch
from torch import nn
import numpy as np


class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=13):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc


class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels.

        An optional flex parameter is added in the forward pass to allow for "x" with different
        numbers of entries in each tile. This occurs when the number of tiles don't perfectly
        divide the number of "pixels" in the data or the data is not sampled uniformly.
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(AdaptiveLinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights)
        torch.nn.init.zeros_(bias)
    
    def forward(self, x, indices):
        """
        if flex != True:
            - x: torch.tensor
            - optimized pytorch code for inference
        else:
            - x: list of torch.tensor. Each tensor in x contains all the entries of a specifi weight tile
            - sequential inference of each weight tile (not optimized)
        """
        return torch.bmm(x, self.weight[indices, ...]) + self.bias[indices, ...]


class AdaptiveReLULayer(nn.Module):
    '''
        Implements ReLU activations with multiple channel input.
        
        The parameters is_first, and omega_0 are not relevant.
    '''
    
    def __init__(self, in_features, out_features, n_channels):
        super().__init__()        
        self.in_features = in_features
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.relu = torch.nn.LeakyReLU(0.2)
        
    def forward(self, input, indices):
        return self.relu(self.linear(input, indices))


class AdaptiveSinLayer(nn.Module):
    '''
        Implements ReLU activations with multiple channel input.
        
        The parameters is_first, and omega_0 are not relevant.
    '''
    
    def __init__(self, in_features, out_features, n_channels):
        super().__init__()        
        self.in_features = in_features
        self.omega_0 = 30.0
        self.const = 1.0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            bound = np.sqrt(self.const*6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, input, indices):
        """
        If flex is False, then the input is a tensor and element-wise relu could be inference 
        directly.

        If flex is True, then the input is a list of tensors. Hence the list operation.
        """
        return torch.sin(self.omega_0 * self.linear(input, indices)) 


class KiloNerf(nn.Module):
    """KiloNerf model."""
    def __init__(
        self,
        in_features,
        out_features,
        data_size,
        kilo_configs
    ):
        super().__init__()
        self.kilo_config = kilo_configs.NET
        self.num_layers = self.kilo_config.num_layers
        self.dim_in = in_features
        self.dim_hidden = self.kilo_config.dim_hidden
        self.dim_out = out_features
        self.num_freq = self.kilo_config.num_freq
        self.data_size = torch.tensor(data_size)
        self.tile_dim = torch.tensor(self.kilo_config.dimensions)
        self.n_tiles = torch.prod(self.tile_dim)

        self.dummy_indices = torch.tensor([idx for idx in range(self.n_tiles)])

        # Network with position dependent weights
        self.net = []
        self.net.append(AdaptiveReLULayer(2*in_features*self.num_freq+in_features, self.dim_hidden, self.n_tiles))
        for i in range(1, self.num_layers-1):
            self.net.append(AdaptiveReLULayer(self.dim_hidden, self.dim_hidden, self.n_tiles))
        self.net.append(AdaptiveLinearWithChannel(self.dim_hidden, out_features, self.n_tiles))
        
        self.net = torch.nn.ModuleList(self.net)

        # Position encoding
        self.pe = PosEncoding(in_features, self.num_freq)

    def reset_weights(self, layer):
        torch.nn.init.kaiming_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    def forward(self, x, labels=None):
        # Tile x based on grid
        x, indices = self.tile_input(x)
        # x should be normalized from [self.data_size[0], self.data_size[1]] to [-1, 1]
        x = (x / self.data_size.to(x.device) - 0.5) * 2
        x = self.pe(x)
        for i, mod in enumerate(self.net):
            # Treat each tile as a separate channel and run inference.
            x = mod(x, self.dummy_indices)

        out = self.untiling(x, indices)
        return out
    
    def tile_input(self, x):
        #x = x * 0.5 + 0.5   # Renormalize x to [0, 1] to utilize algorithm stated in LoE_supplementary
        # Calculate the tile index of the input
        input_tile_index = torch.floor(x / (self.data_size / self.tile_dim).to(x.device)).long()
        #input_tile_index = torch.floor(x * self.data_size.to(x.device)) % self.tile_dim.to(x.device)
        # Convert the axis-dependent tile indices to sequential indices
        # E.g. index: (1, 0, 1) for tile_dim=(2, 2, 2)--> (5)
        input_tile_seq_index = self.coord_to_indices(input_tile_index, self.tile_dim)

        # Add original indices of x for untiling later
        indices = torch.arange(0, len(x), 1).type(torch.int).to(x.device)
        input_tile_seq_index = torch.cat((input_tile_seq_index.unsqueeze(-1), indices.unsqueeze(-1)), dim=1)
        n_tiles = torch.prod(self.tile_dim)
        pixels_per_tile = len(x) // n_tiles
        new_indices = torch.zeros(n_tiles, pixels_per_tile, input_tile_seq_index.shape[-1]).to(x.device)
        
        for tile in range(n_tiles):
            mask = torch.where(input_tile_seq_index[:, 0] == tile, 1, 0).type(torch.bool).to(x.device)
            # Tile indices for later untiling
            new_indices[tile, ...] = input_tile_seq_index[mask]
            # Tile x such that each tile is a "channel" in torch.bmm terminology
            if tile == 0:
                new_x = x[mask, ...].unsqueeze(0)
            else:
                new_x = torch.cat((new_x, x[mask,...].unsqueeze(0)), dim=0)

        return new_x.to(x.device), new_indices.to(x.device)

    def coord_to_indices(self, tile_index, tile_dim):
        """
        Convert the axis-dependent tile indices to sequential indices.
        The process is similar to changing base for a number, where the original base
        is the tile_dim and the values are the axis_indices. The target base is 10.
        E.g.
            - Image dimension: 25x25
            - A coorindate value of (22, 22) for weight tiles of dimensions (5, 5)
            - Axis-dependent tile indices: (4, 4)
            - Sequential indices: 24

        Input:
            - axis_indices: axis dependent tile indices. Shape = (n, dim_in)
            - tile_dim: weight tile dimensions of a single network layer.
        Return:
            - indices: sequential indices of each entry of "x"
        """
        indices = torch.zeros(tile_index.shape[0]).to(tile_index.device)
        for i in range(len(tile_dim)):
            if i == len(tile_dim) - 2:
                indices += tile_index[:, i] * tile_dim[-1]
            elif i == len(tile_dim) - 1:
                indices += tile_index[:, i]
            else:
                indices += tile_index[:, i] * torch.prod(tile_dim[i+1:])
        return indices
    
    def untiling(self, x, indices):
        """
        Untile x back to it's original order.
        This is necessary to maintain input-output correspondence of the network.

        Input:
            - x.shape = (n_tiles, pixels_per_tile, dim_hidden)
            - indices.shape = (n_tiles, pixels_per_tile, num_layer+1)

        indices[:, :, -1] should be the original indices of the pixels
        """
        n_data = x.shape[0] * x.shape[1]
        untiled_x = torch.zeros(n_data, x[0].shape[-1]).to(x[0].device)
        for i, x_slice in enumerate(x):
            untiled_x[indices[i][:, -1].type(torch.int),...] = x_slice

        return untiled_x