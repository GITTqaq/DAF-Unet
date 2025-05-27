import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        # 将输入从torch.Tensor转换为numpy数组
        img1_np = img1.cpu().detach().numpy()
        img2_np = img2.cpu().detach().numpy()
        # 计算SSIM值
        ssim_val = ssim(img1_np.squeeze(), img2_np.squeeze(), data_range=1.0)
        # 将SSIM值转换为损失值
        loss = 1 - ssim_val
        return torch.tensor(loss, requires_grad=True).to(img1.device)

