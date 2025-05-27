import torch
from torchvision.models import vgg16
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision.models.vgg import VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weights = config['loss_weights']  # {'mse':0.5, 'ssim':0.3, 'vgg':0.2}

        # 预加载VGG特征提取器
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        self.vgg = nn.Sequential(*list(vgg.children())[:5]).eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, pred, target):
        pred = pred.to(device)
        target = target.to(device)
        vgg_pred = self.vgg(pred.repeat(1, 3, 1, 1))
        vgg_target = self.vgg(target.repeat(1, 3, 1, 1))
        return F.l1_loss(vgg_pred, vgg_target)

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        ssim_val = 1 - ssim(pred, target, data_range=1.0)
        percep = self.perceptual_loss(pred, target)
        return (self.weights['mse'] * mse +
                self.weights['ssim'] * ssim_val +
                self.weights['vgg'] * percep)