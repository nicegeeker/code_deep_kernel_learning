import torch
import numpy as np

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def warped_grid(model, x, y):
    '''
    To warp grid points, draw transformation of 2D space.
    Parameters:
    -----------------
    model: nn.Model, py torch model for the warp.
    x, y: torch tensor, shape(res, res), generated form meshgrid.

    Returns:
    ------------
    warped_x, warped_y: torch.tensor, shape(res, res), warped x and y , the same shape as input x and y.
    '''
    res = x.shape[0]
    x = x.flatten()
    y = y.flatten()
    x_y = torch.stack([x, y], dim=1)
    warped_x_y = model(x_y)
    warped_x = warped_x_y[:, 0].reshape(res, res)
    warped_y = warped_x_y[:, 1].reshape(res, res)
    return warped_x, warped_y


def mesh_grid_to_polygons(x, y):
    '''
    Change meshgrids to polygons.
    Parameters:
    -----------------
    x, y: torch.tensor, shape(res, res), the output of function ''warped_grid''.

    Returns:
    ------------
    p: PatchCollection, Collection of polygons.
    '''
    if x.device.type == 'cuda':
        x = x.cpu()
        y = y.cpu()

    x = x.detach().numpy()
    y = y.detach().numpy()
    polygons = []
    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]-1):
            polygon = Polygon(np.array([[x[i, j], y[i, j]], [
                              x[i, j+1], y[i, j+1]], [x[i+1, j+1], y[i+1, j+1]], [x[i+1, j], y[i+1, j]]]), True)
            polygons.append(polygon)
    p = PatchCollection(polygons, cmap='viridis')
    return p


def get_color(res):
    color = np.zeros((res-1, res-1))
    for i in range(res-1):
        for j in range(res-1):
            if (i+j) % 2 == 0:
                color[i, j] = 888
            else:
                color[i, j] = 120
    return color.reshape(-1)
