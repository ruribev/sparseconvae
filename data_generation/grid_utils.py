import torch

def create_grid(size):
    y, x = torch.meshgrid(torch.arange(size[0], dtype=torch.float32),
                          torch.arange(size[1], dtype=torch.float32),
                          indexing='ij')
    return torch.stack((y.flatten(), x.flatten()), dim=1).unsqueeze(0)

def create_param_grid(size, params):
    num_params, param_dims = params.shape
    param_grid = torch.zeros(param_dims - 2, size[0], size[1])
    param_mask = torch.zeros(1, size[0], size[1])

    for param in params:
        y, x = param[:2].long()
        if 0 <= y < size[0] and 0 <= x < size[1]:
            param_grid[:, y, x] = param[2:]
            param_mask[0, y, x] = 1

    return param_grid, param_mask
