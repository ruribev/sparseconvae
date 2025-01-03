import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sample(x, mask, param_grid, param_mask, categories=None):
    num_channels = x.shape[0]
    size = x.shape[1]
    fig, axs = plt.subplots(num_channels, 4, figsize=(20, 5*num_channels))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if num_channels == 1:
        axs = [axs]

    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    param_grid = param_grid.cpu().numpy() if isinstance(param_grid, torch.Tensor) else param_grid
    param_mask = param_mask.cpu().numpy() if isinstance(param_mask, torch.Tensor) else param_mask

    for i in range(num_channels):
        im = axs[i][0].imshow(x[i], cmap='viridis', interpolation='nearest')
        axs[i][0].set_title('Spatial Data')
        plt.colorbar(im, ax=axs[i][0])

        mask_indices = np.argwhere(mask[0] > 0)
        mask_values = x[i][mask[0] > 0]
        axs[i][1].imshow(x[i], cmap='viridis', interpolation='nearest', alpha=0.5)
        axs[i][1].scatter(mask_indices[:, 1], mask_indices[:, 0], c=mask_values, cmap='viridis', edgecolors='black', s=50)
        axs[i][1].set_title('Sampling Mask')

        im = axs[i][2].imshow(param_grid[i], cmap='viridis', interpolation='nearest')
        axs[i][2].set_title('Parameter Grid')
        plt.colorbar(im, ax=axs[i][2])

        param_indices = np.argwhere(param_mask[0] > 0)
        param_values = param_grid[i][param_mask[0] > 0]
        axs[i][3].imshow(param_grid[i], cmap='viridis', interpolation='nearest', alpha=0.5)
        axs[i][3].scatter(param_indices[:, 1], param_indices[:, 0], c=param_values, cmap='viridis', edgecolors='black', s=50)
        axs[i][3].set_title('Parameter Mask')

    plt.tight_layout()
    plt.show()
