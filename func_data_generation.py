import torch
from torch import nn
import numpy as np

from math import pi

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from warp_units import *
from func_polygon import *


def recv_power(tran_power, tran_loc, recv_coords, r_η=3.5, fc=700e6):
    '''
    该函数根据tran_loc，节点坐标recv_x, recv_y, 自由空间传播损耗因子η，生成每个节点的接收功率
    Parameters
        ----------
        tran_power: float,  transmitted power in dB
        tran_loc: torch.tensor, shape(2)
        recv_coords: torch.tensor, shape(n, 2), coordinates of ponts, 
        r_η: float,自由空间传播损耗因子
        fc: float, frequency
    Returns
        ----------
        recv_power: torch.tensor, shape(n)    
    '''
    lamda = torch.tensor(3e8/fc)
    recv_signal = tran_power + 20 * torch.log10(lamda/(4 * pi)) - 10 * r_η * torch.log10(
        torch.linalg.norm(recv_coords - tran_loc, dim=1))
    return recv_signal


def chessboard_from_model(model, chessgrid_res=40):
    '''
    Get chessboard transformed by model.
    Parameters:
    -----------------
    chessgrid_res: int, the chess board resolution.

    Returns:
    ------------
    plot the chess board of model warp.
    '''
    local_device = next(model.parameters()).device.type
    x_ = torch.linspace(0, 1, chessgrid_res).to(local_device)
    y_ = torch.linspace(-1, 0, chessgrid_res).to(local_device)
    x, y = torch.meshgrid(x_, y_, )
    warpx, warpy = warped_grid(model, x, y)
    p = mesh_grid_to_polygons(warpx, warpy)
    color = get_color(chessgrid_res)
    fig, ax = plt.subplots(figsize=(6, 6))
    p.set_array(color)
    wx = warpx.cpu()
    wy = warpy.cpu()
    ax.set_xlim([wx.min().detach().numpy(), wx.max().detach().numpy()])
    ax.set_ylim([wy.min().detach().numpy(), wy.max().detach().numpy()])
    ax.add_collection(p)


def oned_map(model, chessgrid_res=40, dim=0):
    '''
    Get chessboard transformed by model.
    Parameters:
    -----------------
    chessgrid_res: int, the chess board resolution.

    Returns:
    ------------
    Plot 1-D map of axial warp model.
    '''
    local_device = next(model.parameters()).device.type
    x_ = torch.linspace(0, 1, chessgrid_res).to(local_device)
    y_ = torch.linspace(-1, 0, chessgrid_res).to(local_device)
    x, y = torch.meshgrid(x_, y_, )
    warpx, warpy = warped_grid(model, x, y)
    if dim == 0:
        x = x.cpu().numpy()[:, 0]
        warpx = warpx.cpu().detach().numpy()[:, 0]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.plot(x, warpx)
    if dim == 1:
        y = y.cpu().numpy()[0, :]
        warpy = warpy.cpu().detach().numpy()[0, :]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.plot(y, warpy)


def get_field_data(resolution, tran_power, tran_loc):
    '''
    Get data of the whole field.
    Parameters:
    -----------------
    resolution: int, the data resolution.
    tran_power: float, transmitted power in dB.
    tran_loc: torch.tensor, shape(1, 2)

    Returns:
    ------------
    coords: torch.tensor, shape(res*res, 2), coordinates in space.
    noshadow: torch.tensor, shape(res*res), receive signal strength only consider free-path loss.
    '''
    step = 1000/resolution
    xs = torch.arange(0, 1000, step=1000 / resolution) + step/2
    ys = torch.flip(torch.arange(-1000, 0, step=1000 /
                    resolution) + step/2, dims=[0])
    xs,  ys = torch.meshgrid(xs, ys)
    xs, ys = xs.T.flatten(), ys.T.flatten()
    coords = torch.stack((xs, ys), dim=1)
    noshadow = recv_power(tran_power=tran_power,
                          tran_loc=tran_loc, recv_coords=coords)

    data = pd.DataFrame(
        {'x': xs.numpy(), 'y': ys.numpy(), 'rss': noshadow.numpy()})
    data = data.pivot('y', 'x', 'rss')
    data = data.sort_index(ascending=False)
#     print(data)
    sns.heatmap(data, cmap='viridis', robust=True,
                square=True, xticklabels=25, yticklabels=25)
#     ax.imshow(noshadow_matrix, cmap='viridis')
#     fig.colorbar(ax=ax)
    return coords, noshadow


def plot_heatmap(coords, data, mask_idx=None):
    '''
    Plot the heatmap of an area.
    Parameters:
    -----------------
    coords: torch.tensor, shape(res*res, 2), coordinates in space, generated from function "get_field_data".
    data:troch.tensor, shape(res*res), the receive signal strength: 1.generated from fnction "get_field_data", 2. sampled form gp_model using coords as training data.
    mask_idx: shape(N), the idx of points to show, generated from function "random_choose".
    '''
    if mask_idx is not None:
        mask_idx = mask_idx.cpu()
        mask = np.ones(data.shape)
        mask = mask.flatten()
        mask[mask_idx] = False
        mask = mask.reshape(100, 100)
    else:
        mask = None

    data = pd.DataFrame({'x': coords[:, 0].cpu().numpy(
    ), 'y': coords[:, 1].cpu().numpy(), 'rss': data.cpu().numpy()})
    data = data.pivot('y', 'x', 'rss')
    data = data.sort_index(ascending=False)
    sns.heatmap(data, cmap='viridis', robust=True, square=True,
                xticklabels=75, yticklabels=75, mask=mask)


def random_choose(coords, data, num=300):
    '''
    Randomly chose points in the whole field for training.
    Parameters:
    -----------------
    coords: torch.tensor, shape(res*res, 2), coordinates in space, generated from function "get_field_data".
    data:troch.tensor, shape(res*res), the receive signal strength: 1.generated from fnction "get_field_data", 2. sampled form gp_model using coords as training data.
    num: int, number of points to choose.

    Returns:
    ------------
    train_X: torch.Tensor, shape(num, 2)
    train_y: torch.Tensor, shape(num)
    idx: the idx of trianing data in the whole field.
    '''
    choose_prob = torch.ones_like(coords[:, 0]) * (1/coords.shape[0])
    idx = choose_prob.multinomial(num, replacement=False)
    train_X = coords[idx, :]
    train_y = data[idx]
    return train_X, train_y, idx
