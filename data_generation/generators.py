import torch
import random
from .grid_utils import create_grid
from .interpolation_methods import *

class BaseModel:
    def __init__(self):
        self.layers = []

    def add_deposit(self, points, value):
        self.layers.append((points, value))
        return self

    def _distribute_values(self, points, value):
        num_points = len(points)
        if isinstance(value, (int, float)):
            return [value] * num_points
        elif isinstance(value, (tuple, list)):
            return np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(value)), value)
        raise ValueError("Value must be a number or an array of numbers")

    def generate_primarys(self):
        primarys = []
        for layer, (points, value) in enumerate(self.layers):
            distributed_values = self._distribute_values(points, value)
            for (x, y), v in zip(points, distributed_values):
                primarys.append((y, x, layer, v))
        return torch.tensor(primarys, dtype=torch.float32)

class SpatialGenerator():
    def __init__(self, size, param_generator, methods=['kriging']):
        self.size = size
        self.param_generator = param_generator
        self.methods = methods if isinstance(methods, list) else [methods]
        self.grid = create_grid(size)

    def generate_item(self):
        # Generate parameters using the provided lambda function
        params = self.param_generator()

        x = torch.zeros(params.shape[1] - 2, self.size[0], self.size[1])  # Assuming params has columns for x, y, and other values
        for i in range(params.shape[1] - 2):
            method_index = min(i, len(self.methods) - 1)
            method = self.methods[method_index].lower().strip()
            channel_params = params[:, [0, 1, i + 2]]
            if method == 'id':
                x[i] = create_id_spatial_data(self.size, channel_params, self.grid)
            elif method == 'vid':
                x[i] = create_vid_spatial_data(self.size, channel_params, self.grid)
            elif method == 'cvid':
                x[i] = create_cvid_spatial_data(self.size, channel_params, self.grid)
            elif method == 'nn':
                x[i] = create_nn_spatial_data(self.size, channel_params, self.grid)
            elif method == 'kriging':
                x[i] = create_kriging_spatial_data(self.size, channel_params, self.grid)
            elif method == 'stationary':
                x[i] = create_stationary_spatial_data(self.size, channel_params, self.grid)
            elif method == 'layers':
                x[i] = create_layered_spatial_data(self.size, channel_params, self.grid)
            elif method == 'mps':
                x[i] = create_mps_spatial_data(self.size, channel_params, self.grid)
            else:
                raise ValueError(f"Unsupported interpolation method: {method}")

        param_grid, param_mask = create_param_grid(self.size, params)

        return x, param_grid, param_mask

class CategoricalSpatialGenerator():
    def __init__(self, size, param_generator, num_categories, methods):
        self.size = size
        self.param_generator = param_generator
        self.num_categories = num_categories
        self.methods = methods
        self.grid = create_grid(size)

    def generate_item(self):
        # Generate parameters using the provided lambda function
        params = self.param_generator()
        params = self.normalize_data(params)

        x = torch.zeros(len(self.methods), self.size[0], self.size[1])
        param_grid = torch.zeros(len(self.methods), self.size[0], self.size[1])
        param_mask = torch.zeros(1, self.size[0], self.size[1])

        # Generate categorical data using the first 3 columns of params
        category_data = self.create_categorical_data(params)
        x[0] = category_data.squeeze()  # Remove the channel dimension
        param_grid[0] = category_data.squeeze()  # Remove the channel dimension

        # Initialize category mask
        for param in params:
            y, x_coord = param[:2].long()
            if 0 <= y < self.size[0] and 0 <= x_coord < self.size[1]:
                param_mask[0, y, x_coord] = 1

        category_params = [[] for _ in range(self.num_categories)]
        for param in params:
            category = self.continuous_to_categorical(param[2]).item()
            category_params[category].append(param)

        for category in range(self.num_categories):
            if len(category_params[category]) > 0:
                cat_params = torch.stack(category_params[category])
                for i, method in enumerate(self.methods[1:], start=1):
                    channel_params = cat_params[:, [0, 1, i + 2]]
                    interpolated_values = self.interpolate(method, channel_params)
                    x[i] = torch.where(category_data.squeeze() == category, interpolated_values, x[i])
                    param_grid[i] = torch.where(category_data.squeeze() == category, interpolated_values, param_grid[i])

        return x, param_grid, param_mask


    def normalize_data(self, params):
        values = params[:, 2:]
        min_val = values.min(0, keepdim=True)[0]
        max_val = values.max(0, keepdim=True)[0]
        normalized_values = (values - min_val) / (max_val - min_val + 1e-6)  # Adding a small constant to avoid division by zero
        params[:, 2:] = normalized_values
        return params

    def create_categorical_data(self, params):
        first_method = self.methods[0]
        continuous_data = self.interpolate_method(first_method, params[:, :3])
        return self.continuous_to_categorical(continuous_data)

    def continuous_to_categorical(self, x):
        return torch.clamp((x * self.num_categories).long(), 0, self.num_categories - 1)

    def interpolate(self, method, channel_params):
        if method == 'id':
            return create_id_spatial_data(self.size, channel_params, self.grid)[0]
        elif method == 'vid':
            return create_vid_spatial_data(self.size, channel_params, self.grid)[0]
        elif method == 'cvid':
            return create_cvid_spatial_data(self.size, channel_params, self.grid)[0]
        elif method == 'nn':
            return create_nn_spatial_data(self.size, channel_params, self.grid)
        elif method == 'kriging':
            return create_kriging_spatial_data(self.size, channel_params, self.grid)
        elif method == 'mps':
            return create_mps_spatial_data(self.size, channel_params, self.grid)
        elif method == 'stationary':
            return create_stationary_spatial_data(self.size, channel_params, self.grid)
        elif method == 'layered':
            return create_layered_spatial_data(self.size, channel_params, self.grid)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def interpolate_method(self, method, channel_params):
        if method == 'id':
            return create_id_spatial_data(self.size, channel_params, self.grid)
        elif method == 'vid':
            return create_vid_spatial_data(self.size, channel_params, self.grid)
        elif method == 'cvid':
            return create_cvid_spatial_data(self.size, channel_params, self.grid)
        elif method == 'nn':
            return create_nn_spatial_data(self.size, channel_params, self.grid)
        elif method == 'kriging':
            return create_kriging_spatial_data(self.size, channel_params, self.grid)
        elif method == 'layered':
            return create_layered_spatial_data(self.size, channel_params, self.grid)
        elif method == 'mps':
            return create_mps_spatial_data(self.size, channel_params, self.grid)
        elif method == 'stationary':
            return create_stationary_spatial_data(self.size, channel_params, self.grid)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")


def generate_points(num_points, width, elevations, noise_factor=0.1):
    x = np.linspace(0, width, num_points)

    # Interpolate elevations
    if len(elevations) > 2:
        control_points = np.linspace(0, width, len(elevations))
        y = np.interp(x, control_points, elevations)
    else:
        y = np.linspace(elevations[0], elevations[-1], num_points)

    # Add noise
    elevation_range = max(elevations) - min(elevations)
    noise = np.random.normal(0, noise_factor * elevation_range, num_points)
    y += noise

    return list(zip(x, y))

def sine_vals(num_values, start=0, amp=40, noise=0.1):
    x = np.linspace(0, 2 * np.pi, num_values)
    amplitude = amp * np.random.rand()
    phase = np.random.rand() * 2 * np.pi
    y = amplitude * np.sin(x + phase) + np.random.normal(0, 0.1, num_values) + start
    return y

def generate_constrained_values(num_values, max_change=0.1):
    base_value = random.random()
    values = [base_value]
    for _ in range(num_values - 1):
        change = random.uniform(-max_change, max_change)
        new_value = max(0, min(1, values[-1] + change))
        values.append(new_value)
    return values

def two_layer_generator():
    model = BaseModel()
    width = 64
    height = 32
    layers = random.randint(2, 2)
    points = random.randint(5, 15)
    amp = height / layers

    # First layer always at 0
    first_layer_points = generate_points(points, width, [0, 0], noise_factor=0)
    random_values = generate_constrained_values(random.randint(2, 5))
    model.add_deposit(first_layer_points, random_values)

    for i in range(1, layers):
        random_values = generate_constrained_values(random.randint(2, 5))
        model.add_deposit(
            generate_points(points, width, sine_vals(points, amp * i - amp/2, amp=amp), noise_factor=0.05),
            random_values
        )

    return model.generate_primarys()

