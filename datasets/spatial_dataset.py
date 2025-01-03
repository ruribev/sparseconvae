import torch
from torch.utils.data import Dataset
import os
import pickle
from datetime import datetime
from tqdm import tqdm


class SpatialDataset(Dataset):
    def __init__(self, num_generations, generator, sampling_fn, secondary_grid_fn, data_folder=None, dynamic_secondary_mask=False, x_channels=None, primary_channels=None, secondary_channels=None):
        self.num_generations = num_generations
        self.generator = generator
        self.sampling_fn = sampling_fn
        self.secondary_grid_fn = secondary_grid_fn
        self.data_folder = data_folder
        self.dynamic_secondary_mask = dynamic_secondary_mask
        self.x_channels = x_channels
        self.primary_channels = primary_channels
        self.secondary_channels = secondary_channels

        if data_folder is not None:
            os.makedirs(self.data_folder, exist_ok=True)
            if num_generations > 0:
                self._generate_and_save_entries()
            self.data = self._load_all_entries()
        else:
            self.data = self._generate_items()

    def __len__(self):
        return len(self.data)

    def _generate_and_save_entries(self):
        entries = self._generate_items()
        with open(os.path.join(self.data_folder, f'entries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'), 'wb') as f:
            pickle.dump(entries, f)

    def _load_all_entries(self):
        entries = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.pkl'):
                with open(os.path.join(self.data_folder, file), 'rb') as f:
                    entries.extend(pickle.load(f))
        return entries

    def _generate_items(self):
        return [self._generate_item() for i in tqdm(range(self.num_generations), desc="Generating items", mininterval=1.0)]

    def _generate_item(self):
        x, primary_grid, primary_mask = self.generator.generate_item()
        secondary_grid = self.secondary_grid_fn(x)
        secondary_mask = self.sampling_fn(secondary_grid.shape[1:])

        return x, primary_grid, primary_mask, secondary_grid, secondary_mask

    def __getitem__(self, idx):
        x, primary_grid, primary_mask, secondary_grid, saved_secondary_mask = self.data[idx]

        if self.dynamic_secondary_mask:
            secondary_mask = self.sampling_fn(secondary_grid.shape[1:])
        else:
            secondary_mask = saved_secondary_mask

        x = self._apply_channel_selection(x, self.x_channels)
        primary_grid = self._apply_channel_selection(primary_grid, self.primary_channels)
        secondary_grid = self._apply_channel_selection(secondary_grid, self.secondary_channels)

        return x, primary_grid, primary_mask, secondary_grid, secondary_mask

    def _apply_channel_selection(self, tensor, channel_selection):
        if channel_selection is not None:
            if isinstance(channel_selection, int):
                return tensor[channel_selection:channel_selection+1]
            elif isinstance(channel_selection, (list, tuple)):
                return tensor[list(channel_selection)]
            else:
                raise ValueError("channel_selection must be an integer, list, or tuple")
        return tensor