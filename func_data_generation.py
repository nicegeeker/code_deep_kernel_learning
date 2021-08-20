import torch
from torch import nn
import numpy as np
import h5py

from math import pi

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from func_polygon import *
from utils import *
from matplotlib import ticker


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


def chessboard_from_model(model, chessgrid_res=40, ax=None):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    p.set_array(color)
    wx = warpx.cpu()
    wy = warpy.cpu()
    ax.set_xlim([wx.min().detach().numpy()-0.025, wx.max().detach().numpy()+0.025])
    ax.set_ylim([wy.min().detach().numpy()-0.025, wy.max().detach().numpy()+0.025])
    ax.add_collection(p)


def oned_map(model, ax=None, chessgrid_res=40, dim=0):
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
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.plot(x, warpx)
    if dim == 1:
        y = y.cpu().numpy()[0, :]
        warpy = warpy.cpu().detach().numpy()[0, :]
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.plot(y, warpy)


def get_field_data(resolution, tran_power, tran_loc, device='cpu', r_η=3.5, fc=700e6):
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
                          tran_loc=tran_loc, recv_coords=coords, r_η=r_η, fc=fc)

    return coords.to(device), noshadow.to(device)


def plot_heatmap(coords, data, mask_idx=None, ax=None, vmin=None, vmax=None, robust=False, cmap='viridis', cbar=False, cbar_ax=None):
    '''
    Plot the heatmap of an area.
    Parameters:
    -----------------
    coords: torch.Tensor or numpy.Array, shape(res*res, 2), coordinates in space, generated from function "get_field_data".
    data:troch.Tensor or numpy.Array, shape(res*res), the receive signal strength: 1.generated from fnction "get_field_data", 2. sampled form gp_model using coords as training data.
    mask_idx: shape(N), the idx of points to show, generated from function "random_choose".
    ax: matplotlib ax object, which ax to draw the heatmap.
    vmin, vmax: anchor of the color bar.
    cbar:bool, draw colore bar or not.
    cbar_ax: which ax to draw the color bar.
    '''
    
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if isinstance(mask_idx, torch.Tensor):
        mask_idx = mask_idx.cpu().numpy()

    shape = np.sqrt(data.shape[0]).astype(int)
    if mask_idx is not None:
        mask = np.ones(data.shape)
        mask = mask.flatten()
        mask[mask_idx] = False
        mask = mask.reshape(shape, shape)
    else:
        mask = None


    data = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'rss': data})
    data = data.pivot('y', 'x', 'rss')
    data = data.sort_index(ascending=False)

    sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, robust=robust, square=True,
                mask=mask, ax=ax, cbar=cbar, cbar_ax=cbar_ax)
    tick_locator = ticker.LinearLocator(numticks=6)

    def xformat_func(value, pos):
        return '%d' % (value / shape *1000)

    def yformat_func(value, pos):
        return '%d' % (-value / shape * 1000)

    xtick_formatter = ticker.FuncFormatter(xformat_func)
    ytick_formatter = ticker.FuncFormatter(yformat_func)
    ax.xaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_locator(tick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)
    ax.yaxis.set_major_formatter(ytick_formatter)


def random_choose(coords, data, num=300, seed=100):
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
    torch.manual_seed(seed)
    idx = choose_prob.multinomial(num, replacement=False)
    train_X = coords[idx, :]
    train_y = data[idx]
    return train_X, train_y, idx


def data_generation(gp_model, likelihood, grid_res=100, tran_power=30, seed=0, sigma_noise=0.2):
    '''
    Generate data from gp_model.
    Draw two figure: true_rss , train_y
    Store result in data floder with name_train_X, name_train_y, name_coords, name_true_rss.

    Parameter:
    ----------
    gp_model: GP model in CUDA.
    likehood: Likelihood model in CUDA.
    grid_res: int, Grid resolution for field data.
    tran_power: float, Transmmited power of signal source.
    seed: int, Random seed for generating signal source location.
    choose_num: int, Traning datasets size.

    Returns:
    --------
    train_X: torch.Tensor, 
    train_y:
    idx: 
    coords:
    true_rss:
    '''
    torch.manual_seed(seed)
    tran_x = torch.rand(()) * 1000
    tran_y = -torch.rand(()) * 1000
    tran_loc = torch.stack((tran_x, tran_y), dim=-1)
    device = next(gp_model.parameters()).device
    coords, free_path = get_field_data(grid_res, tran_power, tran_loc, device)

    coords = coords.to(device)

    likelihood.eval()
    gp_model.eval()
    # GP model space is from 0 to 1
    observed_pred = likelihood(gp_model(coords/1000))
    shadow = observed_pred.sample()

    noise = torch.normal(0, sigma_noise, size=free_path.shape).to(device)
    true_rss = free_path + shadow
    observed_rss = free_path + shadow + noise

    return {'coords': coords,
            'true_rss':true_rss,
            'observed_rss':observed_rss,
            'free_path':free_path,
            'shadow':shadow,
            'noise':noise,
            } 


def save_data(file_name, datasets):
    """[Save generated data as hdf5.]

    Args:
        file_name ([string]): [File name without extension.]
        datasets ([tuple]): [Returns of function data_generation.]
    """
    file = DATA_PATH + '/' + file_name + '.hd5f'
    with h5py.File(file, 'w') as f:
        f['coords'] = datasets['coords'].cpu().numpy()
        f['true_rss'] = datasets['true_rss'].cpu().numpy()
        f['observed_rss'] = datasets['observed_rss'].cpu().numpy()
        f['free_path'] = datasets['free_path'].cpu().numpy()
        f['shadow'] = datasets['shadow'].cpu().numpy()
        f['noise'] = datasets['noise'].cpu().numpy()

        
def random_choose_subtract(train_X, train_y, idx, num=50, seed=100):
    '''
    Randomly chose N points in the dataset and subtract them from datasets, return the rest.
    Parameters:
    -----------------
    train_X: torch.tensor, shape(N, 2), coordinates in space.
    train_y:troch.tensor, shape(N), the receive signal strength.
    idx: idx of the orignal.
    num: int, number of points to subtract.
    seed: random seed.

    Returns:
    ------------
    train_X: torch.Tensor, shape(N-num, 2)
    train_y: torch.Tensor, shape(N-num)
    '''
    N = train_X.shape[0]
    choose_prob = torch.ones_like(train_X[:, 0]) * (1/train_X.shape[0])
    torch.manual_seed(seed)
    idx_temp = choose_prob.multinomial(N-num, replacement=False)
    train_X_new = train_X[idx_temp, :]
    train_y_new = train_y[idx_temp]
    idx_new = idx[idx_temp]
    return train_X_new, train_y_new, idx_new
    
    
def add_training_data(file_name, min_num=200, max_num=2000, step=50):
    '''
    Choose samples in datasets to generate training data.
    '''
    file = DATA_PATH + '/' + file_name + '.hd5f'
    if os.path.exists(file):
        with h5py.File(file, "r") as f1:
            coords = f1["coords"][...]
            observed_rss = f1["observed_rss"][...]
            shadow_noise = f1["shadow"][...] + f1["noise"]
        coords = torch.from_numpy(coords)
        shadow_noise = torch.from_numpy(shadow_noise)
        
    num = max_num 
    train_X, train_y, idx = random_choose(coords, shadow_noise, num=num)
    while num >= min_num:
        train_X_numpy = train_X.numpy()
        train_y_numpy = train_y.numpy()
        idx_numpy = idx.cpu().numpy()
        with h5py.File(file, 'a') as f:
            subgroup = f.require_group('train_' + str(num))
            subgroup.require_dataset('train_X', train_X_numpy.shape, train_y_numpy.dtype)
            subgroup['train_X'][...] = train_X_numpy
            
            subgroup.require_dataset('train_y', train_y_numpy.shape, train_y_numpy.dtype)
            subgroup['train_y'][...] = train_y_numpy
            
            subgroup.require_dataset('idx', idx_numpy.shape, idx_numpy.dtype)
            subgroup['idx'][...] = idx_numpy
        num -= step
        train_X, train_y, idx = random_choose_subtract(train_X, train_y, idx, num=step)
        
        
def save_result(file_name, model_name, sensor_num, predict_mean, predict_var=None):
    '''
    Save GP model's prediction into hd5f file.
    
    Parameters:
    -----------
    file_name: string, file name with no extension.
    model_name: string, name of GP model(gp, dgp, warpgp...)
    sensor_num: int, number of sensors.
    predict_mean: numpy array, mean of result.
    predict_var: numpy array, variance of result.
    
    file structrue:
    file------gp------50------predict_mean
           
    
    '''
    file = DATA_PATH + '/' + file_name + '.hd5f'
    with h5py.File(file, 'a') as f:
        model_group = f.require_group(model_name)
        sensor_num_group = model_group.require_group(str(sensor_num))
        
        
        shape = predict_mean.shape
        dtype = predict_mean.dtype
            
        sensor_num_group.require_dataset('predict_mean', shape, dtype)
        sensor_num_group['predict_mean'][...] = predict_mean
        if predict_var is not None:
            sensor_num_group.require_dataset('predict_var', shape, dtype)
            sensor_num_group['predict_var'][...] = predict_var
            
            
def load_data(data_num, sensor_num):
    data_filename = DATA_PATH + "/datasets_" + str(data_num) + ".hd5f"

    if os.path.exists(data_filename):
        with h5py.File(data_filename, "r") as f1:
            coords = f1["coords"][...]
            shadow = f1["shadow"][...]
            shadow_noise = f1["shadow"][...] + f1["noise"][...]
            train_X = f1["train_" + str(sensor_num) + "/train_X"][...]
            train_y = f1["train_" + str(sensor_num) + "/train_y"][...]
            idx = f1["train_" + str(sensor_num) + "/idx"][...]
    
    return coords, shadow, shadow_noise, train_X, train_y, idx
    