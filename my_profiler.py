import torch


def taco_profiler(n_layers, dim_in, hidden_dims, dim_out, dimensions, data_shape):
    assert n_layers == len(dimensions[0])
    assert n_layers == len(hidden_dims) + 1

    data_size = torch.prod(torch.tensor(data_shape))
    dimensions = torch.tensor(dimensions)
    MAC = 0
    # MAC of backbone model
    for i in range(n_layers):
        n_tiles = torch.prod(dimensions[:, :i+1])
        # MAC of layer
        if i == 0:
            module_MAC = dim_in * hidden_dims[0]
        elif i == n_layers - 1:
            module_MAC = hidden_dims[i-1] * dim_out
        else:
            module_MAC = hidden_dims[i-1] * hidden_dims[i]
        MAC += module_MAC * n_tiles / data_size

    # MAC of modulators
    for i in range(n_layers - 1):
        n_tiles = torch.prod(dimensions[:, i+1])
        # MAC of modulator
        modulator_MAC = dim_in * hidden_dims[i]
        MAC += modulator_MAC * n_tiles / data_size

    return MAC


def asmr_profiler(n_layers, dim_in, hidden_dims, dim_out, dimensions, data_shape, mod='linear'):
    data_size = torch.prod(torch.tensor(data_shape))
    dimensions = torch.tensor(dimensions)
    MAC = 0
    # MAC of backbone model
    for i in range(n_layers):
        n_tiles = torch.prod(dimensions[:, :i+1])
        # MAC of layer
        if i == 0:
            module_MAC = dim_in * hidden_dims[0] + hidden_dims[0]
        elif i == n_layers - 1:
            module_MAC = hidden_dims[i-1] * dim_out + dim_out
        else:
            module_MAC = hidden_dims[i-1] * hidden_dims[i] + hidden_dims[i]
        MAC += module_MAC * n_tiles / data_size

    # MAC of modulators
    for i in range(n_layers - 1):
        n_tiles = torch.prod(dimensions[:, i+1])
        # MAC of modulator
        # linear MAC
        if mod == 'linear':
            modulator_MAC = dim_in * hidden_dims[i]
        # nonlinear MAC (3 layer, 32 hidden dim, SIREN)
        elif mod == 'nonlinear':
            hidden_dim = 32
            layers = 3
            modulator_MAC = dim_in * hidden_dim + hidden_dim
            for _ in range(layers-2):
                modulator_MAC += hidden_dim * hidden_dim + hidden_dim
            modulator_MAC += hidden_dim * hidden_dims[i] + hidden_dims[i]
        MAC += modulator_MAC * n_tiles / data_size
        # mac of adding modulators to backbone
        MAC += hidden_dims[i] * torch.prod(dimensions[:, :i+1]) / data_size

    return MAC.item()


def mlp_profiler(n_layers, dim_in, hidden_dims, dim_out):
    assert n_layers == len(hidden_dims) + 1

    MAC = 0
    # MAC of backbone model
    for i in range(n_layers):
        # MAC of layer
        if i == 0:
            module_MAC = dim_in * hidden_dims[0] + hidden_dims[0]
        elif i == n_layers - 1:
            module_MAC = hidden_dims[i-1] * dim_out + dim_out
        else:
            module_MAC = hidden_dims[i-1] * hidden_dims[i] + hidden_dims[i]
        MAC += module_MAC

    return MAC


if __name__ == "__main__":
    # ULTRA-LOW MAC
    n_layers = 7
    hidden_dims = [256] * (n_layers-1)
    asmr1_dimensions = [[2,2,2,2,2,2,8], [2,2,2,2,2,2,8]]
    asmr2_dimensions = [[4,2,2,2,2,8], [4,2,2,2,2,8]]
    asmr3_dimensions = [[8,2,2,2,8], [8,2,2,2,8]]
    asmr4_dimensions = [[4,4,4,8], [4,4,4,8]]
    asmr5_dimensions = [[8,8,8], [8,8,8]]
    
    dim_in = 2
    dim_out = 1

    data_shape = (512, 512)
    asmr1_mac = asmr_profiler(n_layers, dim_in, hidden_dims, dim_out, asmr1_dimensions, data_shape, mod='linear')
    asmr2_mac = asmr_profiler(6, dim_in, [256]*5, dim_out, asmr2_dimensions, data_shape, mod='linear')
    asmr3_mac = asmr_profiler(5, dim_in, [256]*4, dim_out, asmr3_dimensions, data_shape, mod='linear')
    asmr4_mac = asmr_profiler(4, dim_in, [256]*3, dim_out, asmr4_dimensions, data_shape, mod='linear')
    asmr5_mac = asmr_profiler(3, dim_in, [256]*2, dim_out, asmr5_dimensions, data_shape, mod='linear')
    mlp_mac = mlp_profiler(n_layers, dim_in, hidden_dims, dim_out)
    print("ULTRA LOW MAC")
    print("====================================")
    print("ASMR-1 mac: %s (%sx)" % (int(asmr1_mac), int(mlp_mac/asmr1_mac)))
    print("ASMR-2 mac: %s (%sx)" % (int(asmr2_mac), int(mlp_mac/asmr2_mac)))
    print("ASMR-3 mac: %s (%sx)" % (int(asmr3_mac), int(mlp_mac/asmr3_mac)))
    print("ASMR-4 mac: %s (%sx)" % (int(asmr4_mac), int(mlp_mac/asmr4_mac)))
    print("ASMR-5 mac: %s (%sx)" % (int(asmr5_mac), int(mlp_mac/asmr5_mac)))
    print("MLP mac: ", mlp_mac)
    print("====================================")

    # MEGAPIXEL
    n_layers = 6
    hidden_dims = [1024] * (n_layers-1)
    taco_dimensions = [[8,4,4,4,16], [8,4,4,4,16]]
    asmr1_dimensions = [[4,4,4,4,4,8], [4,4,4,4,4,8]]
    asmr2_dimensions = [[4,4,4,4,2,16], [4,4,4,4,2,16]]
    
    dim_in = 2
    dim_out = 3

    data_shape = (8192, 8192)
    ngp_mac = mlp_profiler(2, 16, [64], dim_out)
    taco_mac = taco_profiler(5, dim_in, [1024]*4, dim_out, taco_dimensions, data_shape)
    asmr1_mac = asmr_profiler(n_layers, dim_in, hidden_dims, dim_out, asmr1_dimensions, data_shape, mod='nonlinear')
    asmr2_mac = asmr_profiler(n_layers, dim_in, hidden_dims, dim_out, asmr2_dimensions, data_shape, mod='nonlinear')
    mlp_mac = mlp_profiler(n_layers, dim_in, hidden_dims, dim_out)
    print("MEGAPIXEL MAC")
    print("====================================")
    print("NGP mac: %s (%sx)" % (int(ngp_mac), int(mlp_mac/ngp_mac)))
    print("TACO mac: %s (%sx)" % (int(taco_mac), int(mlp_mac/taco_mac)))
    print("ASMR-1 mac: %s (%sx)" % (int(asmr1_mac), int(mlp_mac/asmr1_mac)))
    print("ASMR-2 mac: %s (%sx)" % (int(asmr2_mac), int(mlp_mac/asmr2_mac)))
    print("MLP mac: ", mlp_mac)
    print("====================================")

    # VIDEO (UVG)
    n_layers = 10
    hidden_dims = [512] * (n_layers-1)
    asmr_dimensions = [[5, 1, 1, 5, 1, 1, 3, 1, 1, 2], [5, 1, 3, 1, 3, 1, 3, 1, 2, 1], [5, 3, 2, 2, 1, 2, 1, 2, 1, 2]]
    
    dim_in = 3
    dim_out = 3
    data_shape = (150, 270, 480)
    asmr_mac = asmr_profiler(n_layers, dim_in, hidden_dims, dim_out, asmr_dimensions, data_shape, mod='linear')
    kilo_mac = mlp_profiler(6, dim_in, [64]*5, dim_out)
    mlp_mac = mlp_profiler(n_layers, dim_in, hidden_dims, dim_out)
    print("VIDEO MAC")
    print("====================================")
    print("ASMR mac: %s (%sx)" % (int(asmr_mac), int(mlp_mac/asmr_mac)))
    print("KILO mac: %s (%sx)" % (int(kilo_mac), int(mlp_mac/kilo_mac)))
    print("MLP mac: ", mlp_mac)
    print("====================================")

    # VIDEO (CAT)
    n_layers = 7
    hidden_dims = [512] * (n_layers-1)
    asmr2_dimensions = [[5,3,1,2,1,2,5], [4,2,2,2,2,2,4], [4,2,2,2,2,2,4]]
    asmr3_dimensions = [[5,5,3,2,1,1,2], [4,4,2,2,2,2,2], [4,4,2,2,2,2,2]]
    
    dim_in = 3
    dim_out = 3
    data_shape = (300, 512, 512)
    asmr2_mac = asmr_profiler(7, dim_in, hidden_dims, dim_out, asmr2_dimensions, data_shape, mod='linear')
    asmr3_mac = asmr_profiler(7, dim_in, hidden_dims, dim_out, asmr3_dimensions, data_shape, mod='linear')
    mlp_mac = mlp_profiler(n_layers, dim_in, hidden_dims, dim_out)
    print("VIDEO (CAT) MAC")
    print("====================================")
    print("ASMR-2 mac: %s (%sx)" % (int(asmr2_mac), int(mlp_mac/asmr2_mac)))
    print("ASMR-3 mac: %s (%sx)" % (int(asmr3_mac), int(mlp_mac/asmr3_mac)))
    print("MLP-2 mac: ", mlp_mac)
    print("====================================")

