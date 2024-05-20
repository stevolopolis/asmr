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


class AdaptiveSinLayer(nn.Module):
    '''
        Implements ReLU activations with multiple channel input.
        
        The parameters is_first, and omega_0 are not relevant.
    '''
    
    def __init__(self, in_features, out_features, n_channels, omega_0=30.0, is_first=False):
        super().__init__()        
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.const = 1.0
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = self.const/self.in_features
                self.linear.weight.uniform_(-bound, bound)      
                self.linear.bias.uniform_(-bound, bound)
            else:
                bound = np.sqrt(self.const*6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, input, indices, flex=False):
        """
        If flex is False, then the input is a tensor and element-wise relu could be inference 
        directly.

        If flex is True, then the input is a list of tensors. Hence the list operation.
        """
        if not flex:
            return torch.sin(self.omega_0 * self.linear(input, indices, flex=flex)) 
        else:
            out = self.omega_0 * self.linear(input, indices, flex=flex)
            return [torch.sin(out[i]) for i in range(len(out))]


class LoE(nn.Module):
    """LoE model.

    The range of x must be [-1, 1]

    To implement the position-dependent weights, we go through the following preprocessing steps:
        1. Obtain the weight indices of all layers for each entry of x.
        2. Obtain the masks for each weight tile in each layer.
        3. Tile the input data based on the masks of each weight tile.
        4. Treat each tile as a separate channel and run inference on each layer.
        5. Untile the output of each layer back to it's original order.
        6. Run inference on the last layer.
    
    Note that there are two implemeentations:
        1. (When # of data for each weight tile is consistent) - Tensorized input and optimized inference
        2. (When # of data is flexible) - List of tensors input and sequential of each weight tile
    """

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

    def forward(self, x, labels=None, flex=True):
        x = (x / torch.tensor(labels).to(x) - 0.5) * 2
        # Weight tile indices and masks for input data are only calculated once per model initialization
        if self.indices is None or x.shape[0] != self.n_data:
            self.get_tile_indices(x, flex)

        # x should be normalized to be within [-1, 1]
        x = self.pe(x)
        for i, mod in enumerate(self.net):
            # Tile the input data based on the masks of each weight tile
            # x.shape = (n_tiles, pixels_per_tile, ...)
            if flex:
                x = self.tile_input_flex(x, self.tile_dims[i], self.masks[i])
            else:
                x = self.tile_input(x, self.tile_dims[i], self.masks[i])
            # Treat each tile as a separate channel and run inference.
            x = mod(x, self.dummy_indices[i], flex=flex)

        x = self.untiling(x, self.indices, self.n_data)
        return self.last_layer(x)
    
    def get_tile_indices(self, x, flex):
        self.masks = []     # Masks of input data corresponding to each weight tile for each layer
        self.dummy_indices = []
        self.n_data = x.shape[0]
        # Get weight indices of all layers for each entry of x
        self.indices = self.get_indices(x, self.tile_dims, img_shape=self.img_shape)
        
        # Tile the weight indices based on the output shape of each layer; and,
        # Get the tile masks for each layer
        for i in range(len(self.net)):
            if flex:
                self.indices, tile_masks = self.tile_indices_flex(self.n_data, self.indices, self.tile_dims[i], i, x.device)
            else:
                self.indices, tile_masks = self.tile_indices(self.n_data, self.indices, self.tile_dims[i], i, x.device)
            self.masks.append(tile_masks)
            self.dummy_indices.append(torch.tensor([idx for idx in range(torch.prod(self.tile_dims[i]))]).to(x.device))
    
    def get_indices(self, x, tile_dims, img_shape=None):
        """
        Get weight indices of all layers for each entry of the input "x". 

        Input:
            - x: normalized coordinates. Shape = [n, dim_in]
            - tile_dims: weight tile dimensions of each network layer. E.g. [[2,2], [2,2], [2,2]]
        Return:
            - weight_indices: weight indices of all layers for each entry of the input "x". Shape = [n, num_layers+1]
        
        Note that the last column of "weight_indices" is the original indices of the pixels.
        This is necessarily saved to allow us to revert the order of inferenced data back to
            the original input order, such that input-output correspondence is maintained.
        """
        x = (x - 0.5) * 2   # Renormalize x to [0, 1] to utilize algorithm stated in LoE_supplementary
        tile_dims = tile_dims.to(x.device)

        # Tensor to keep tract of cumulative product of tile dimensions.
        cumprod_tile_dims = torch.tensor([1 for _ in range(x.shape[1])]).to(x.device)

        weight_indices = None
        for tile_dim in tile_dims:
            cumprod_tile_dims *= tile_dim

            # Axis-dependent tile indices for each entry of "x"
            axis_layer_indices = torch.floor(x * cumprod_tile_dims) % tile_dim
            # Convert the axis-dependent tile indices to sequential indices
            # E.g. index: (1, 0, 1) for tile_dim=(2, 2, 2)--> (5)
            layer_indices = self.coord_to_indices(axis_layer_indices, tile_dim)
            # Concatenate the weight indices of each layer
            if weight_indices is None:
                weight_indices = layer_indices.unsqueeze(-1)
            else:
                weight_indices = torch.cat((weight_indices, layer_indices.unsqueeze(-1)), dim=1)
        
        if img_shape is not None:
            indices = torch.arange(0, len(x), 1).type(torch.int).to(x.device)
            weight_indices = torch.cat((weight_indices, indices.unsqueeze(-1)), dim=1)

        return weight_indices.to(x.device)

    def coord_to_indices(self, axis_indices, tile_dim):
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
        indices = torch.zeros(axis_indices.shape[0]).to(axis_indices.device)
        for i in range(len(tile_dim)):
            if i == len(tile_dim) - 2:
                indices += axis_indices[:, i] * tile_dim[-1]
            elif i == len(tile_dim) - 1:
                indices += axis_indices[:, i]
            else:
                indices += axis_indices[:, i] * torch.prod(tile_dim[i+1:])
        return indices


    def tile_indices(self, n_data, indices, tile_dim, layer, device):
        """
        Generate the mask for each weight tile in the current <layer>.

        Input:
            - n_data: total number of data points
            - indices: sequential indices of each entry of "x". Shape = (n_data, n_layers)
            - tile_dim: weight tile dimensions of a single network layer.
            - layer: current layer
            - device: device of the network and data. E.g. "cuda:0"
        Return:
            - new_indices: reshaped input <indices> such that for each i, new_indices[i] 
                contains the indices of the pixels that belong to the same weight tile in
                the current <layer>.
            - full_mask: mask for each weight tile in the current <layer>. Shape = (n_tiles, n_data/n_tiles)
        """
        n_tiles = torch.prod(tile_dim)
        pixels_per_tile = int(n_data / n_tiles)
        new_indices = torch.zeros(n_tiles, pixels_per_tile, indices.shape[-1]).to(device)

        # efficient version
        for tile in range(n_tiles):
            mask = torch.where(indices[..., layer] == tile, 1, 0).type(torch.bool).to(device)
            new_indices[tile, ...] = indices[mask]
            if layer == 0:
                mask = mask.unsqueeze(0)
            if tile == 0:
                full_mask = mask.unsqueeze(0)
            else:
                full_mask = torch.cat((full_mask, mask.unsqueeze(0)), dim=0)

        return new_indices, full_mask

    def tile_indices_with_padding(self, n_data, indices, tile_dim, layer, device):
        """
        Generate the mask for each weight tile in the current <layer>.

        Input:
            - n_data: total number of data points
            - indices: sequential indices of each entry of "x". Shape = (n_data, n_layers)
            - tile_dim: weight tile dimensions of a single network layer.
            - layer: current layer
            - device: device of the network and data. E.g. "cuda:0"
        Return:
            - new_indices: reshaped input <indices> such that for each i, new_indices[i] 
                contains the indices of the pixels that belong to the same weight tile in
                the current <layer>.
            - full_mask: mask for each weight tile in the current <layer>. Shape = (n_tiles, n_data/n_tiles)
        """
        n_tiles = torch.prod(tile_dim)

        # Calculate max dim
        max_rows = 0
        for tile in range(n_tiles):
            mask = torch.where(indices[..., layer] == tile, 1, 0).type(torch.bool).to(device)
            n_rows = torch.sum(mask).item()
            if n_rows > max_rows:
                max_rows = n_rows
        new_indices = torch.zeros(n_tiles, max_rows, indices.shape[-1]).to(device)

        # efficient version
        for tile in range(n_tiles):
            mask = torch.where(indices[..., layer] == tile, 1, 0).type(torch.bool).to(device)
            padding = torch.zeros(max_rows - torch.sum(mask).item(), indices.shape[-1]).to(device)
            new_indices[tile, ...] = indices[mask] + padding
            if layer == 0:
                mask = mask.unsqueeze(0)
            if tile == 0:
                full_mask = mask.unsqueeze(0)
            else:
                full_mask = torch.cat((full_mask, mask.unsqueeze(0)), dim=0)

        return new_indices, full_mask


    def tile_indices_flex(self, n_data, indices, tile_dim, layer, device):
        """
        Generate the mask for each weight tile in the current <layer>.
        This flexible version allows for different number of entries in each weight tile.
        As a consequence, a list of tensors is returned instead of a single tensor.

        Input:
            - n_data: total number of data points
            - indices: sequential indices of each entry of "x". Shape = (n_data, n_layers)
            - tile_dim: weight tile dimensions of a single network layer.
            - layer: current layer
            - device: device of the network and data. E.g. "cuda:0"
        Return:
            - new_indices: reshaped input <indices> such that for each i, new_indices[i] 
                contains the indices of the pixels that belong to the same weight tile in
                the current <layer>.
            - full_mask: mask for each weight tile in the current <layer>. Shape = (n_tiles, n_data/n_tiles)
        """
        n_tiles = torch.prod(tile_dim)
        new_indices = []
        full_mask = []
        if type(indices) is not list:
            indices = indices.unsqueeze(0)

        for tile in range(n_tiles):
            mask = []
            for i, idx_slice in enumerate(indices):
                mask_slice = torch.where(idx_slice[:, layer] == tile, 1, 0).type(torch.bool).to(device)
                if i == 0:
                    new_idx = idx_slice[mask_slice, ...]
                else:
                    new_idx = torch.cat((new_idx, idx_slice[mask_slice, ...]), dim=0)
                mask.append(mask_slice)
            full_mask.append(mask)
            new_indices.append(new_idx)
            
        return new_indices, full_mask


    def tile_input(self, x, tile_dim, masks):
        """
        Tile input data <x> based on the masks of each weight tile.
        
        Input:
            - x: input data.
                if <x> has not been tiled (i.e. first layer). Shape = (n_data, dim_in)
                else. Shape = (n_tiles, pixels_per_tile, dim_in)
            - tile_dim: weight tile dimensions of a single network layer.
            - masks: mask for each weight tile in the current <layer>.
                if <x> has not been tiled (i.e. first layer). Shape = (n_tiles, n_data)
                else. Shape = (n_tiles, n_tiles, pixels_per_tile)
        Return:
            - new_x: tiled <x>. Shape (n_tiles, pixels_per_tile, dim_in)
        """
        # If x has not been tiled
        n_tiles = torch.prod(tile_dim)
        for i in range(n_tiles):
            if i == 0:
                new_x = x[masks[i], ...].unsqueeze(0)
            else:
                new_x = torch.cat((new_x, x[masks[i],...].unsqueeze(0)), dim=0)

        return new_x


    def tile_input_flex(self, x, tile_dim, masks):
        """
        Tile input data <x> based on the masks of each weight tile.
        This flexible version allows for different number of entries in each weight tile.
        As a consequence, we have to iterate through the list of subsets of <x> that
            corresponds to each weight tile to operate the masking.
        
        Input:
            - x: input data.
                if <x> has not been tiled (i.e. first layer). Shape = (n_data, dim_in)
                else. Shape = (n_tiles, pixels_per_tile, dim_in)
            - tile_dim: weight tile dimensions of a single network layer.
            - masks: mask for each weight tile in the current <layer>.
                if <x> has not been tiled (i.e. first layer). Shape = (n_tiles, n_data)
                else. Shape = (n_tiles, n_tiles, pixels_per_tile)
        Return:
            - new_x: tiled <x>. Shape (n_tiles, pixels_per_tile, dim_in)
        """
        n_tiles = torch.prod(tile_dim)
        new_x = []
        for i in range(n_tiles):
            for j, mask in enumerate(masks[i]):
                if j == 0:
                    x_tmp = x[j][mask, ...]
                else:
                    x_tmp = torch.cat((x_tmp, x[j][mask,...]), dim=0)
            new_x.append(x_tmp)

        return new_x

    def untiling(self, x, indices, n_data):
        """
        Untile x back to it's original order.
        This is necessary to maintain input-output correspondence of the network.

        Input:
            - x.shape = (n_tiles, pixels_per_tile, dim_hidden)
            - indices.shape = (n_tiles, pixels_per_tile, num_layer+1)

        indices[:, :, -1] should be the original indices of the pixels
        """
        untiled_x = torch.zeros(n_data, x[0].shape[-1]).to(x[0].device)
        for i, x_slice in enumerate(x):
            untiled_x[indices[i][:, -1].type(torch.int),...] = x_slice

        return untiled_x

