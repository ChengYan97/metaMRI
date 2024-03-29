"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

#def psnr_torch(target_volume, predicted_volume, max_val):
#    '''
#    PSNR chould be computed per volume and then averaged over volumes.
#    target_volume and predicted_volume must have dimensions [1, num_slices, w, h]
#    '''
#    mse = torch.mean((target_volume-predicted_volume)**2)
#    psnr = 20* torch.log10(torch.tensor(max_val, dtype=mse.dtype, device=mse.device)) - 10 * torch.log10(mse)
#    return psnr

#class PSNRLoss(nn.Module):  
#    '''
#    PSNR chould be computed per volume and then averaged over volumes.
#    target_volume and predicted_volume must have dimensions [1, num_slices, w, h]
#    '''  
#    def __init__(self):
#        super().__init__()#
#
#    def forward(self, target_volume: torch.Tensor, predicted_volume: torch.Tensor, max_val):
#        mse = torch.mean((target_volume-predicted_volume)**2)
#        psnr = 20* torch.log10(torch.tensor(max_val, dtype=mse.dtype, device=mse.device)) - 10 * torch.log10(mse)
#        return psnr

#class L1Loss(torch.nn.Module):
#    def __init__(self):
#        super().__init__()#
#
#    def forward(self,x,y):
#        loss = F.l1_loss(y - x, reduction='sum') / F.l1_loss(x.detach(), reduction='sum')
#        return loss

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.win_size = win_size
        self.k1, self.k2 = torch.tensor(k1).to(device), torch.tensor(k2).to(device)
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size).to(device) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = torch.tensor(NP / (NP - 1)).to(device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
