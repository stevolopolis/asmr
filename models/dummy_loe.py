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
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(1, coords.shape[0], self.out_dim)


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
        self.bias = torch.nn.Parameter(torch.zeros(1,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights)
        torch.nn.init.zeros_(bias)
    
    def forward(self, x, indices, flex=False):
        """
        if flex != True:
            - x: torch.tensor
            - optimized pytorch code for inference
        else:
            - x: list of torch.tensor. Each tensor in x contains all the entries of a specifi weight tile
            - sequential inference of each weight tile (not optimized)
        """
        if not flex:
            return torch.bmm(x, self.weight[indices, ...]) + self.bias[0, ...]
        else:
            out = []
            for i in range(len(indices)):
                tmp = torch.bmm(x[i].unsqueeze(0), self.weight[torch.tensor(indices[i]).unsqueeze(0),...]) + self.bias[0,...]
                out.append(tmp.squeeze(0))
            return out


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
        
    def forward(self, input, indices, flex=False):
        """
        If flex is False, then the input is a tensor and element-wise relu could be inference 
        directly.

        If flex is True, then the input is a list of tensors. Hence the list operation.
        """
        if not flex:
            return self.relu(self.linear(input, indices, flex=flex)) 
        else:
            out = self.linear(input, indices, flex=flex)
            return [self.relu(out[i]) for i in range(len(out))]



class LoE(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        loe_configs
    ):
        super().__init__()
        self.loe_configs = loe_configs.NET
        self.dim_in = in_features
        self.dim_hidden = self.loe_configs.dim_hidden
        self.dim_out = out_features
        self.num_freq = self.loe_configs.num_freq
        self.indices = None
        self.img_shape = self.loe_configs.img_dim
        self.tile_dims = torch.tensor(list(self.loe_configs.dimensions))
        self.masks = []
        self.dummy_indices = []
        self.n_data = 0

        # Load up dummy indices
        for i in range(len(self.tile_dims)):
            self.dummy_indices.append(torch.tensor([idx for idx in range(torch.prod(self.tile_dims[i]))]))

        # Network with position dependent weights
        self.net = []
        self.net.append(AdaptiveReLULayer(2*in_features*self.num_freq+in_features, self.dim_hidden, torch.prod(self.tile_dims[0])))
        for i in range(1, len(self.tile_dims)):
            self.net.append(AdaptiveReLULayer(self.dim_hidden, self.dim_hidden, torch.prod(self.tile_dims[i])))
        
        self.net = torch.nn.ModuleList(self.net)

        # Position encoding
        self.pe = PosEncoding(in_features, self.num_freq)
        # Final layer
        self.last_layer = torch.nn.Linear(self.dim_hidden, out_features, bias=True)
        self.reset_weights(self.last_layer)

    def reset_weights(self, layer):
        torch.nn.init.kaiming_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    def forward(self, x, labels=None):
        # x should be normalized to be within [-1, 1]
        x = self.pe(x)
        for i, mod in enumerate(self.net):
            n_tiles = torch.prod(self.tile_dims[i], dtype=torch.int)
            x = torch.reshape(x, (n_tiles, -1, x.shape[-1]))
            x = mod(x, self.dummy_indices[i])

        out = torch.reshape(x, (-1, self.dim_hidden))
        return self.last_layer(out)
