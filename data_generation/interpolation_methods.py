import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

def create_id_spatial_data(size, params, grid, exponent=2.0):
    coords = params[:, :2]
    values = params[:, 2]
    distances = torch.cdist(grid, coords.unsqueeze(0)).squeeze(0)  # [size[0]*size[1], num_params]
    weights = 1 / (distances ** exponent + 1e-6)
    weighted_values = (weights * values).sum(dim=1)
    weights_sum = weights.sum(dim=1)
    interpolated_values = weighted_values / weights_sum

    return interpolated_values.reshape(1, size[0], size[1])

def create_stationary_spatial_data(size, params, grid):
    mean = params[:, 2].mean().item()
    variance = 0.1

    # Ensure grid is a NumPy array with the correct shape
    if isinstance(grid, torch.Tensor):
        grid = grid.squeeze(0).cpu().numpy()

    # Ensure params is also a NumPy array
    if isinstance(params, torch.Tensor):
        params = params.cpu().numpy()

    # Extract coordinates and known values
    coords = params[:, :2]
    known_values = params[:, 2]

    def exponential_covariance(h, range=3):
        return np.exp(-h / range)

    # Calculate pairwise distances within the entire grid for covariance
    distances = pdist(grid)
    cov_matrix = exponential_covariance(squareform(distances))

    # Find closest grid points to known coordinates
    distances_to_known = cdist(grid, coords)
    closest_indices = np.argmin(distances_to_known, axis=0)

    # Extract necessary sub-matrices from the covariance matrix
    C_oo = cov_matrix[np.ix_(closest_indices, closest_indices)]
    C_ou = cov_matrix[closest_indices, :]
    C_uu = cov_matrix

    # Conditional mean and covariance for Gaussian field
    conditional_mean = mean + C_ou.T @ inv(C_oo) @ (known_values - mean)
    conditional_cov = C_uu - C_ou.T @ inv(C_oo) @ C_ou

    z_conditional = np.random.multivariate_normal(conditional_mean, variance * conditional_cov)
    z_conditional = 1 / (1 + np.exp(-z_conditional))  # Logistic transformation
    z_conditional = np.clip(z_conditional, 0, 1)  # Clipping to ensure [0, 1]

    # Convert the result to a PyTorch tensor
    return torch.from_numpy(z_conditional.reshape((1, size[0], size[1]))).float()

def create_vid_spatial_data(size, params, grid, exponent=2.0):
    height, width = size

    # Extract x and value from params
    x_coords = params[:, 1].numpy()  # Use x-coordinate (column 1)
    values = params[:, 2].numpy()    # Use the first value (column 2)

    # Sort the points by x-coordinate
    sort_idx = np.argsort(x_coords)
    x_coords = x_coords[sort_idx]
    values = values[sort_idx]

    # Create interpolation function
    f = interp1d(x_coords, values, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Generate x values for interpolation
    x_range = np.arange(width)

    # Interpolate values
    interpolated_values = f(x_range)

    # Create 2D output by repeating the interpolated values vertically
    output = np.tile(interpolated_values, (height, 1))

    return torch.tensor(output, dtype=torch.float32).unsqueeze(0)

def create_cvid_spatial_data(size, params, grid, exponent=2.0):
    height, width = size

    # Extract x and value from params
    x_coords = params[:, 1].numpy()  # Use x-coordinate (column 1)
    values = params[:, 2].numpy()    # Use the first value (column 2)

    # Sort the points by x-coordinate
    sort_idx = np.argsort(x_coords)
    x_coords = x_coords[sort_idx]
    values = values[sort_idx]

    # Create interpolation function
    f = interp1d(x_coords, values, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Generate x values for interpolation
    x_range = np.arange(width)

    # Interpolate values
    interpolated_values = f(x_range)

    # Create 2D output by repeating the interpolated values vertically
    output = np.tile(interpolated_values, (height, 1))

    # Round values to nearest 0.1
    output = np.round(output * 10) / 10

    return torch.tensor(output, dtype=torch.float32).unsqueeze(0)

def create_nn_spatial_data(size, params, grid):
    distances = torch.cdist(grid, params[:, :2].unsqueeze(0)).squeeze(0)  # [size[0]*size[1], num_params]
    nearest_idx = distances.argmin(dim=1)  # Get indices of the nearest parameters
    values = params[nearest_idx, 2]  # Use the value of the nearest parameter
    return values.reshape(1, size[0], size[1])

def create_kriging_spatial_data(size, params, grid, range_param=100.0, nugget=1e-5):
    # Extract coordinates and values
    coords = params[:, :2].cpu().numpy()
    values = params[:, 2].cpu().numpy()

    # Calculate distance matrix
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # Calculate covariance matrix (using an exponential model)
    variogram_model = lambda h: np.exp(-h / range_param)
    cov_matrix = variogram_model(dist_matrix)

    # Add nugget effect
    cov_matrix += np.eye(len(cov_matrix)) * nugget

    # Calculate weights
    weights = np.linalg.solve(cov_matrix, values)

    # Calculate distances from grid points to known points
    grid_coords = grid[0].cpu().numpy()
    dist_to_grid = cdist(grid_coords, coords, metric='euclidean')

    # Calculate covariance from grid points to known points
    cov_to_grid = variogram_model(dist_to_grid)

    # Interpolate values at grid points
    interpolated_values = cov_to_grid @ weights

    return torch.tensor(interpolated_values.reshape(size[0], size[1]), dtype=torch.float32).unsqueeze(0)


def create_layered_spatial_data(size, params, grid):
    height, width = size
    output = torch.full((1, height, width), 0.0)  # Start with all zeros

    # Group points by their integer value (layer)
    layers = {}
    for y, x, value in params:
        layer = int(value*10)
        if layer not in layers:
            layers[layer] = []
        layers[layer].append((x.item(), y.item(), value.item()))

    # Sort layers by value (bottom to top)
    sorted_layers = sorted(layers.items(), key=lambda x: x[0])

    # Keep track of the highest point filled for each x-coordinate
    highest_filled = torch.zeros(width, dtype=torch.long)

    # Create splines for each layer
    for layer_value, points in sorted_layers:
        if len(points) < 2:
            continue  # Need at least 2 points for interpolation
        points.sort(key=lambda p: p[0])  # Sort points by x-coordinate
        x_coords, y_coords, values = zip(*points)

        # Create a linear interpolation
        spline = interp1d(x_coords, y_coords, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Evaluate each pixel for this layer
        for x in range(width):
            y_intersect = int(spline(x))
            y_start = max(y_intersect, highest_filled[x])
            if y_start < height:
                output[0, y_start:, x] = layer_value / 10
                highest_filled[x] = y_start

    return output

def create_mps_spatial_data(size, params, grid):
    height, width = size

    # Initialize MPSlib with GENESIM method
    O = mps.mpslib(method='mps_genesim', simulation_grid_size=np.array([ height, width, 1]), verbose_level=0, debug_level=-1)

    # Set up MPSlib parameters
    O.parameter_filename = 'mps.txt'
    O.par['n_real'] = 1
    O.par['n_cond'] = 25
    O.par['template_size'] = np.array([[10, 5], [10, 5], [1, 1]])

    value =  (params[:, 2] < 0.5)
    print(value)
    # Prepare hard data
    hard_data = np.column_stack((params[:, 0], params[:, 1], np.zeros(params.shape[0]), value))
    O.d_hard = hard_data

    # Generate or load a training image
    TI, _ = mps.trainingimages.strebelle(di=2, coarse3d=1)
    O.ti = TI

    # Run the simulation
    O.run()

    # Get the simulated result
    sim = O.sim[0].squeeze()

    return torch.tensor(sim, dtype=torch.float32).unsqueeze(0)