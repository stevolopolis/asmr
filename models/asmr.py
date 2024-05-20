from einops import rearrange, repeat
import torch
import torch.nn as nn
import numpy as np
from .siren import Siren
import logging
import math

log = logging.getLogger(__name__)

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.w0 = w0

    def forward(self, x):
        return self.w0*self.linear(x)


class Modulator_nonlinear(nn.Module):
    def __init__(self, dim_in, dim_out, mod_layers=3, mod_hidden=32, w0=30.): 
        from easydict import EasyDict
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        siren_config = EasyDict({"NET": {"num_layers": mod_layers,
                                         "dim_hidden": mod_hidden,
                                         "w0":w0,
                                         "w0_initial": w0,
                                         "use_bias": True}})
        self.model = Siren(dim_in, dim_out, siren_config)

    def forward(self, x):
        return self.model(x)


class ASMR(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        data_size,
        asmr_configs
    ):
        super().__init__()
        self.asmr_configs = asmr_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        num_layers = self.asmr_configs.num_layers
        dim_hidden = self.asmr_configs.dim_hidden
        w0 = self.asmr_configs.w0

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.data_size = data_size
        self.dimensions = self.asmr_configs.dimensions # list of lists
        self.shared_inference = self.asmr_configs.shared_inference
        self.random_partition = False

        assert all(self.data_size[i] == math.prod(self.dimensions[i]) for i in range(len(self.data_size))), "The product of dimensions along each axis should equal the size of data along that axis"
        assert all(len(dim) == len(self.dimensions[0]) for dim in self.dimensions), "The no. of resolution levels along each axis should match"

        self.dimension_cumprods = [np.cumprod(self.dimensions[i]) for i in range(self.dim_in)]
        self.n_levels = len(self.dimensions[0])
    
        if self.asmr_configs.modulator == "linear":
            self.modulators = nn.ModuleList([Modulator(dim_in, dim_hidden, w0=w0) for i in range(self.n_levels-1)])
        elif self.asmr_configs.modulator == "nonlinear":
            self.modulators = nn.ModuleList([Modulator_nonlinear(dim_in, dim_hidden, w0=w0) for i in range(self.n_levels-1)])
        else:
            raise ValueError("Wrong modulator specifications. Please input ('linear' / 'nonlinear')")

        assert self.n_levels == num_layers, "The no. of resolution levels should match the no. of layers"

        self.siren = Siren(
            dim_in=dim_in,
            dim_out=dim_out,
            siren_configs=asmr_configs
        )        

        self.net = self.siren.net
        self.last_layer = self.siren.last_layer

        log.info("Partition: " + str(self.dimensions))
        self.partition_coords = []
        for i in range(self.n_levels):
            grid = [torch.linspace(0., 1., self.dimensions[j][i]) for j in range(self.dim_in)]
            self.partition_coords.append(torch.stack(torch.meshgrid(grid), dim=-1).view(-1, self.dim_in))

    def decompose_coords(self, coords):
        d = len(self.data_size)
        cum_bases = [self.data_size[i] // np.cumprod(self.dimensions[i]) for i in range(d)]
        ori_coords = [coords[..., i].unsqueeze(1) for i in range(d)]
        level_coords = []
        for l in range(self.n_levels):
            coords_l = [torch.floor_divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] for i in range(d)]
            level_coords.append(torch.cat(coords_l, dim=-1))

        return torch.stack(level_coords, dim=1)                 

    def normalize_coords(self, coords):
        coords_max = (torch.tensor(self.dimensions).T - 1).unsqueeze(0).to(coords.device)
        coords_max[coords_max == 0] = 1
        coords /= coords_max # [0, 1]
        coords[coords != coords] = 0       # Remove Nan when certain coords at certain partition is 1 (i.e. max=0)

        return coords

    def shared_forward(self, x, labels=None):
        # ASMR with hierarchical forward to save MACs
        modulations = [self.modulators[i-1](self.partition_coords[i].to(x.device)) for i in range(1, self.n_levels)]
        x = self.partition_coords[0].to(x.device)
        for i, module in enumerate(self.net):
            x = module.linear(x)
            if self.dim_in == 1:
                x  = repeat(x, 't c -> (t t2) c', t2=self.dimensions[0][i+1])
                x = x + repeat(modulations[i], 't c -> (t2 t) c', t2=self.dimension_cumprods[0][i])
            elif self.dim_in == 2:
                x = rearrange(x, '(h w) c -> h w c', h=self.dimension_cumprods[0][i], w=self.dimension_cumprods[1][i])
                x = repeat(x, 'h w c -> (h h2) (w w2) c', h2=self.dimensions[0][i+1], w2=self.dimensions[1][i+1])  # upsample
                shift = rearrange(modulations[i], '(h w) c -> h w c', h=self.dimensions[0][i+1], w=self.dimensions[1][i+1])
                # x = torch.add(x, repeat(shift, 'h w c -> (h2 h) (w2 w) c', h2=self.dimension_cumprods[0][i], w2=self.dimension_cumprods[1][i]))
                x = x + repeat(shift, 'h w c -> (h2 h) (w2 w) c', h2=self.dimension_cumprods[0][i], w2=self.dimension_cumprods[1][i]) # repeat h2 times along h axis, w2 times along w axis
                x = rearrange(x, 'h w c -> (h w) c')
            elif self.dim_in == 3:
                x = rearrange(x, '(t h w) c -> t h w c', t=self.dimension_cumprods[0][i], h=self.dimension_cumprods[1][i], w=self.dimension_cumprods[2][i])
                x = repeat(x, 't h w c -> (t t2) (h h2) (w w2) c', t2=self.dimensions[0][i+1], h2=self.dimensions[1][i+1], w2=self.dimensions[2][i+1])
                shift = rearrange(modulations[i], '(t h w) c -> t h w c', t=self.dimensions[0][i+1], h=self.dimensions[1][i+1], w=self.dimensions[2][i+1])
                x = x + repeat(shift, 't h w c -> (t t2) (h2 h) (w2 w) c', t2=self.dimension_cumprods[0][i], h2=self.dimension_cumprods[1][i], w2=self.dimension_cumprods[2][i])
                x = rearrange(x, 't h w c -> (t h w) c')
            else:
                raise NotImplementedError("only support 1d, 2d & 3d data")
            x = module.activation(x)
         
        out = self.last_layer(x)
        return out

    def normal_forward(self, x, labels=None):
        x_levels = self.decompose_coords(x) # [b, hw, n_levels, c]
        x_levels = self.normalize_coords(x_levels)
        x = x_levels[..., 0, :] # first level coords
        for i, module in enumerate(self.net):
            x = module.linear(x)
            x = x + self.modulators[i](x_levels[..., i+1, :])
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        out = self.last_layer(x)
        return out

    def forward(self, x, labels=None):
        if self.shared_inference:
            return self.shared_forward(x)
        return self.normal_forward(x)


class ASMRExp(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        data_size,
        asmr_configs
    ):
        super().__init__()
        self.asmr_configs = asmr_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        dim_hidden = self.asmr_configs.dim_hidden
        w0 = self.asmr_configs.w0

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.data_size = data_size
        self.dimensions = self.asmr_configs.dimensions # list of lists
        self.random_partition = False

        assert all(len(dim) == len(self.dimensions[0]) for dim in self.dimensions), "The no. of resolution levels along each axis should match"

        self.n_levels = len(self.dimensions[0])
        if self.asmr_configs.modulator == "linear":
            self.modulators = nn.ModuleList([Modulator(dim_in, dim_hidden, w0=w0) for i in range(self.n_levels-1)])
        elif self.asmr_configs.modulator == "nonlinear":
            self.modulators = nn.ModuleList([Modulator_nonlinear(dim_in, dim_hidden, w0=w0) for i in range(self.n_levels-1)])
        else:
            raise ValueError("Wrong modulator specifications. Please input ('linear' / 'nonlinear')")


        self.siren = Siren(
            dim_in=dim_in,
            dim_out=dim_out,
            siren_configs=asmr_configs
        )        

        self.net = self.siren.net
        self.last_layer = self.siren.last_layer

        if self.asmr_configs.lora:
            self.load_backbone(self.asmr_configs.pretrain_path)
            self.freeze_backbone()
            #self.zero_init_modulators()
            
        log.info("Partition: " + str(self.dimensions))
        self.partition_coords = []
        for i in range(self.n_levels):
            grid = [torch.linspace(0., 1., self.dimensions[j][i]) for j in range(self.dim_in)]
            self.partition_coords.append(torch.stack(torch.meshgrid(grid), dim=-1).view(-1, self.dim_in))

    def load_backbone(self, weights_path):
        "Load backbone weights from a pretrained SIREN."
        print("Loading pretrained SIREN backbone weights...")
        # Load SIREN state_dict
        state_dict = torch.load(weights_path)
        self.siren.load_state_dict(state_dict)

    def zero_init_modulators(self):
        def zero_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
        
        self.modulators.apply(zero_init)

    def freeze_backbone(self):
        for params in self.net.parameters():
            params.requires_grad = False
        for params in self.last_layer.parameters():
            params.requires_grad = False

    def unfreeze_backbone(self):
        for params in self.net.parameters():
            params.requires_grad = True
        for params in self.last_layer.parameters():
            params.requires_grad = True

    def update_backbone(self, iter):
        if iter > 9999:
            self.unfreeze_backbone()

    def decompose_coords(self, coords):
        d = len(self.data_size)
        cum_bases = [self.data_size[i] // np.cumprod(self.dimensions[i]) for i in range(d)]
        ori_coords = [coords[..., i].unsqueeze(1) for i in range(d)]
        level_coords = []
        for l in range(self.n_levels):
            coords_l = [torch.floor_divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] / self.dimensions[i][l] for i in range(d)] # With normalization
            level_coords.append(torch.cat(coords_l, dim=-1))

        return torch.stack(level_coords, dim=1)                 

    def normalize_coords(self, coords):
        coords_max = (torch.tensor(self.dimensions).T - 1).unsqueeze(0).to(coords.device)
        coords_max[coords_max == 0] = 1
        coords /= coords_max # [0, 1]
        coords[coords != coords] = 0       # Remove Nan when certain coords at certain partition is 1 (i.e. max=0)
        # Normalize from [0, 1] to [-1, 1]
        coords -= 0.5
        coords *= 2
        return coords

    def shared_forward(self, x, labels=None):
        # ASMR with hierarchical forward to save MACs
        modulations = [self.modulators[i-1](self.partition_coords[i].to(x.device)) for i in range(1, self.n_levels)]
        x = self.partition_coords[0].to(x.device)
        for i, module in enumerate(self.net):
            x = module.linear(x)
            if self.dim_in == 1:
                x  = repeat(x, 't c -> (t t2) c', t2=self.dimensions[0][i+1])
                x = x + repeat(modulations[i], 't c -> (t2 t) c', t2=self.dimension_cumprods[0][i])
            elif self.dim_in == 2:
                x = rearrange(x, '(h w) c -> h w c', h=self.dimension_cumprods[0][i], w=self.dimension_cumprods[1][i])
                x = repeat(x, 'h w c -> (h h2) (w w2) c', h2=self.dimensions[0][i+1], w2=self.dimensions[1][i+1])  # upsample
                shift = rearrange(modulations[i], '(h w) c -> h w c', h=self.dimensions[0][i+1], w=self.dimensions[1][i+1])
                # x = torch.add(x, repeat(shift, 'h w c -> (h2 h) (w2 w) c', h2=self.dimension_cumprods[0][i], w2=self.dimension_cumprods[1][i]))
                x = x + repeat(shift, 'h w c -> (h2 h) (w2 w) c', h2=self.dimension_cumprods[0][i], w2=self.dimension_cumprods[1][i]) # repeat h2 times along h axis, w2 times along w axis
                x = rearrange(x, 'h w c -> (h w) c')
            elif self.dim_in == 3:
                x = rearrange(x, '(t h w) c -> t h w c', t=self.dimension_cumprods[0][i], h=self.dimension_cumprods[1][i], w=self.dimension_cumprods[2][i])
                x = repeat(x, 't h w c -> (t t2) (h h2) (w w2) c', t2=self.dimensions[0][i+1], h2=self.dimensions[1][i+1], w2=self.dimensions[2][i+1])
                shift = rearrange(modulations[i], '(t h w) c -> t h w c', t=self.dimensions[0][i+1], h=self.dimensions[1][i+1], w=self.dimensions[2][i+1])
                x = x + repeat(shift, 't h w c -> (t t2) (h2 h) (w2 w) c', t2=self.dimension_cumprods[0][i], h2=self.dimension_cumprods[1][i], w2=self.dimension_cumprods[2][i])
                x = rearrange(x, 't h w c -> (t h w) c')
            else:
                raise NotImplementedError("only support 1d, 2d & 3d data")
            x = module.activation(x)
         
        out = self.last_layer(x)
        return out

    def normal_forward(self, x, labels=None):
        x_levels = self.decompose_coords(x) # [b, hw, n_levels, c]
        x_levels = self.normalize_coords(x_levels)
        x = x_levels[..., 0, :] # first level coords
        for i, module in enumerate(self.net):
            x = module.linear(x)
            x = x + self.modulators[i](x_levels[..., i+1, :])
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        out = self.last_layer(x)
        return out

    def forward(self, x, labels=None):
        if self.shared_inference:
            return self.shared_forward(x)
        return self.normal_forward(x)
