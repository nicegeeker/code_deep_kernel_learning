
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class AxialWarpUnit(nn.Module):
    def __init__(self, dim=0, res=50,  grad=200, lims=[0, 1.0], **kwargs):
        '''
        Define an axial warp unit
        Args:
        -------
        dim: int, 0 or 1,  which dimension of axial want to warp, 0 -->x, 1--> y
        res: int,  reselution of warp
        grad:  grad of sigmoid function
        lim: range for warp
        '''
        super().__init__(**kwargs)

        # useful variables , determined value for warp
        self.dim = dim
        llim = torch.tensor(lims[0])
        rlim = torch.tensor(lims[1])
        theta_grad = torch.ones((res-1, 1)) * torch.tensor(grad)
        theta_locs = torch.linspace(llim, rlim, steps=(
            res-1) + 2)[1:-1].reshape(res-1, 1)
        self.register_buffer('llim', llim)
        self.register_buffer('rlim', rlim)
        self.register_buffer('theta_grad', theta_grad)
        self.register_buffer('theta_locs', theta_locs)

        # parameter
        self.weight_true = nn.Parameter(torch.randn((1, res)))

    def forward(self, x):
        x_to_warp = x[:, self.dim]
        min__ = self.warp_f(self.llim)
        max__ = self.warp_f(self.rlim)
        if self.dim == 0:
            x_to_warp = (self.warp_f(x_to_warp) - min__) / (max__ - min__)
            return torch.stack((x_to_warp.flatten(), x[:, 1]), dim=1)
        if self.dim == 1:
            x_to_warp = (self.warp_f(x_to_warp) - min__) / \
                (max__ - min__) - 1.0
            return torch.stack((x[:, 0], x_to_warp.flatten()), dim=1)

    def warp_f(self, x):
        sigmoid_vec = 1.0 / \
            (1.0 + torch.exp(- self.theta_grad * (x - self.theta_locs)))
        warp_vec = torch.cat((x.reshape(1, -1), sigmoid_vec), dim=0)
        out = torch.matmul(torch.exp(self.weight_true), warp_vec)
        return out


class RBFWarpUnit(nn.Module):
    def __init__(self, a, r_0, r_1, **kwargs):
        '''
        Define a RBF warp unit.
        Args:
        -------
        a:
        r_0, r_1: center of RBF warp.
        '''
        super().__init__(**kwargs)

        # useful variables
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('r', torch.tensor([r_0, r_1]))

        # parameter no constrain
        self.weight_true = nn.Parameter(torch.randn(()))

    def forward(self, x):
        temp = 1.0 + torch.exp(torch.tensor(3.0/2.0))/2.0
        weight_constrained = temp / (1.0 + torch.exp(-self.weight_true)) - 1.0
        warped_x = x + weight_constrained * \
            (x - self.r)*torch.exp(-self.a *
                                   torch.square(torch.norm(x-self.r, p=2, dim=1, keepdim=True)))
#         warped_x = x + self.weight_true*(x - self.r)*torch.exp(-self.a*torch.square(torch.norm(x-self.r, p=2, dim=1, keepdim=True)))
        return warped_x


def rbfunit_sequential(res=1, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
    '''
    Generate a sequential model for RBF wraping.
    pamameters:
    -----------
    res: int, resolution of RBF wraping is (3^res) * (3^res).
    lim: list, Limitation of the field.

    retruns:
    -----------
    model: a sequential RBF warp model for the whole field.
    '''
    rx = np.linspace(xlim[0], xlim[1], 3**res)
    ry = np.linspace(ylim[0], ylim[1], 3**res)
    r_0s, r_1s = np.meshgrid(rx, ry)
    r_0s = r_0s.flatten()
    r_1s = r_1s.flatten()
    a = (2*(3**res - 1))**2
    model = nn.Sequential()
    i = 0
    for r_0, r_1 in zip(r_0s, r_1s):
        model.add_module('rbf_unit_res'+str(res)+'num' +
                         str(i), RBFWarpUnit(a, r_0, r_1))
        i += 1
    return model


class MobiusWarpUnit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Parameters
        self.a1Re = nn.Parameter(torch.randn(()))
        self.a2Re = nn.Parameter(torch.randn(()))
        self.a3Re = nn.Parameter(torch.randn(()))
        self.a4Re = nn.Parameter(torch.randn(()))

        self.a1Im = nn.Parameter(torch.randn(()))
        self.a2Im = nn.Parameter(torch.randn(()))
        self.a3Im = nn.Parameter(torch.randn(()))
        self.a4Im = nn.Parameter(torch.randn(()))

    def forward(self, x):
        a1 = torch.complex(real=self.a1Re, imag=self.a1Im)
        a2 = torch.complex(real=self.a2Re, imag=self.a2Im)
        a3 = torch.complex(real=self.a3Re, imag=self.a3Im)
        a4 = torch.complex(real=self.a4Re, imag=self.a4Im)

        z = torch.complex(real=x[:, 0], imag=x[:, 1])
        p = (a1 * z + a2) / (a3*z + a4)

        return torch.stack([torch.real(p), torch.imag(p)], dim=1)
