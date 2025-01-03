import torch
import random

def random_sampling(size):
    num_points = int(0.01 * size[0] * size[1])  
    return (torch.rand(1, *size) < (num_points / (size[0] * size[1]))).float()

# Grid sampling pattern
def grid_sampling(size):
    num_points = int(0.1 * size[0] * size[1]) 
    step = max(1, int(min(size) / (num_points ** 0.5)))
    mask = torch.zeros(1, *size)
    mask[0, ::step, ::step] = 1
    return mask

# Clustered sampling pattern
def clustered_sampling(size):
    num_points = int(0.1 * size[0] * size[1])  
    mask = torch.zeros(1, *size)
    num_clusters = 5
    points_per_cluster = num_points // num_clusters
    for _ in range(num_clusters):
        center = torch.randint(0, min(size), (2,))
        cluster_points = (torch.randn(points_per_cluster, 2) * (min(size) / 10) + center).long().clamp(0, size[0] - 1, 0, size[1] - 1)
        mask[0, cluster_points[:, 0], cluster_points[:, 1]] = 1
    return mask


def drilling_sampling(size, min_drillholes=5, max_drillholes=15, min_samples=3, max_samples=20):
    mask = torch.zeros(1, *size)
    height, width = size
    num_drillholes = random.randint(min_drillholes, max_drillholes)

    for _ in range(num_drillholes):
        x = random.randint(0, width - 1)
        num_samples = random.randint(min_samples, max_samples)
        y_positions = random.sample(range(height), min(num_samples, height))
        for y in y_positions:
            mask[0, y, x] = 1