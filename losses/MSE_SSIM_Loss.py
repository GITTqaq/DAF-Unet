import torch
import torch.nn as nn
from torch.nn import functional as F

class MSE_SSIM_Loss(nn.Module):
    def __init__(self, mse_weight=0.6, ssim_weight=0.4, window_size=11, sigma=1.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.window = None  # 先初始化为 None
        self.window_size = window_size
        self.sigma = sigma

    def _gaussian_kernel(self, device):
        coords = torch.arange(self.window_size).float()
        coords -= (self.window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).view(1, 1, self.window_size, self.window_size)  # [1,1,H,W]
        window = window.to(device)  # 移动到指定设备
        return window

    def ssim(self, img1, img2, data_range=1.0):
        if self.window is None or self.window.device != img1.device:
            self.window = self._gaussian_kernel(img1.device)

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu1 = F.conv2d(img1, self.window, padding=0)
        mu2 = F.conv2d(img2, self.window, padding=0)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=0) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=0) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)

        return ssim_map.mean()

    def forward(self, pred, target):
        # 输入约束：pred/tensor需在[0,1]范围
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)

        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss