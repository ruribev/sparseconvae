
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from scipy.interpolate import interp1d
import numpy as np


import numpy as np
import torch
import matplotlib.pyplot as plt
import random

import os, random, torch, numpy as np
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm.notebook import tqdm



"""## Sparse Autoencoder Model"""


"""# Train"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(sparse_encoder, imputation_decoder, subsurface_decoder, dataloader, num_epochs=10, learning_rate=1e-4, device="cuda"):
    sparse_encoder.train()
    imputation_decoder.train()
    subsurface_decoder.train()

    optimizer = optim.AdamW(list(sparse_encoder.parameters()) + list(imputation_decoder.parameters())+ list(subsurface_decoder.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_parameter_loss = 0
        total_sample_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, primary_grid, primary_mask, secondary_grid, secondary_mask = batch
            batch_x = x.to(device)
            batch_primary_grid = primary_grid.to(device)
            batch_primary_mask = primary_mask.to(device)
            batch_secondary_grid = secondary_grid.to(device).float()
            batch_secondary_mask = secondary_mask.to(device)

            optimizer.zero_grad()

            global _cur_active
            _cur_active = batch_secondary_mask

            features = sparse_encoder(batch_secondary_grid*batch_secondary_mask)
            sample_output = imputation_decoder(features[::-1])
            parameter_output = subsurface_decoder(features[::-1])

            sample_loss = criterion(sample_output, batch_secondary_grid)
            parameter_loss = criterion(parameter_output, batch_x)

            total_sample_loss += sample_loss.item()
            total_parameter_loss += parameter_loss.item()

            loss = sample_loss + parameter_loss

            loss.backward()
            optimizer.step()

            num_batches += 1

        avg_sample_loss = total_sample_loss / num_batches
        avg_parameter_loss = total_parameter_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Sample Loss: {avg_sample_loss:.10f}, Average Parameter Loss: {avg_parameter_loss:.10f}")

        with torch.no_grad():
            visualize_results(batch_secondary_grid[0], batch_secondary_mask[0], sample_output[0])
            visualize_results(batch_x[0], batch_primary_mask[0], parameter_output[0])

    return sparse_encoder, imputation_decoder, subsurface_decoder

"""# Train Sequence"""

def drilling_sampling_custom(size, min_drillholes=5, max_drillholes=15, min_samples=3, max_samples=20):
    mask = torch.zeros(1, *size)
    height, width = size
    num_drillholes = random.randint(min_drillholes, max_drillholes)

    for _ in range(num_drillholes):
        x = random.randint(0, width - 1)
        num_samples = random.randint(min_samples, max_samples)
        y_positions = random.sample(range(height), min(num_samples, height))
        for y in y_positions:
            mask[0, y, x] = 1

    return mask

dataset = SpatialDataset(0, generator, drilling_sampling_custom, secondary_grid_fn, data_folder="drive/MyDrive/data_res/two_layers", dynamic_secondary_mask=True, x_channels=1, secondary_channels=0, primary_channels=1)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

convnext = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
sparse_encoder = SparseEncoder(convnext, input_size=(32, 64)).to(device)
imputation_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768, sbn=False).to(device)
subsurface_decoder = Decoder(up_sample_ratio=32, out_chans=1, width=768, sbn=False).to(device)

sparse_encoder, imputation_decoder, subsurface_decoder = train_model(sparse_encoder, imputation_decoder, subsurface_decoder, dataloader, num_epochs=100, device=device)