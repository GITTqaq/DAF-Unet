import torch
import torch.nn as nn
from aaainfrared_image_fusion.models.components import DynamicAttentionFusion,FrequencyEnhancement,NonLocalBlock,MultiScaleConv
class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#
# class DepthwiseSeparableConv(nn.Module):              #深度可卷积分离
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         mid_channels = mid_channels or out_channels
#         self.double_conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, mid_channels),
#             DepthwiseSeparableConv(mid_channels, out_channels)
#         )

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            # self.attention = DynamicAttentionFusion(in_channels // 2)#                        加入注意力机制！！！！！！！！！！
            # self.freq_enhance = FrequencyEnhancement(in_channels // 2)
            # self.non_local = NonLocalBlock(in_channels // 2, downsample_factor=0.5)                        #降噪+残差

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 = self.attention(x2)                                 #                        加入注意力机制！！！！！！！！！！
        # x2 = self.freq_enhance(x2)
        # x2 = self.non_local(x2)                   #降噪
        x = torch.cat([x2, x1], dim=1)  # 直接拼接，假设尺寸已对齐
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        # self.inc = MultiScaleConv(n_channels, 64)  #多尺度卷积块
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # 解码器
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 解码器
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)  # 输出 [0, 1]
        # return logits