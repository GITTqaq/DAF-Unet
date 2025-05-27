import torch
import torch.nn as nn


class DynamicAttentionFusion(nn.Module):  #通道注意力
    def __init__(self, in_channels):
        super(DynamicAttentionFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#
# class DynamicAttentionFusion(nn.Module): #空间注意力+通道注意力
#     def __init__(self, in_channels):
#         super(DynamicAttentionFusion, self).__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 16, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 16, in_channels, 1),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 通道注意力
#         channel_att = self.channel_attention(x)
#         x = x * channel_att
#         # 空间注意力
#         spatial_att = self.spatial_attention(torch.cat([torch.max(x, dim=1, keepdim=True)[0],
#                                                       torch.mean(x, dim=1, keepdim=True)], dim=1))
#         return x * spatial_att

#
# class DynamicAttentionFusion(nn.Module):  #空间注意力
#     """空间注意力模块"""
#     def __init__(self, in_channels, kernel_sizes=[3, 5], reduction=16):
#         super(DynamicAttentionFusion, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(2, in_channels // reduction, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, 1, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 计算最大池化和均值池化特征
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         mean_pool = torch.mean(x, dim=1, keepdim=True)
#         # 拼接池化特征
#         pool_features = torch.cat([max_pool, mean_pool], dim=1)
#         # 生成空间注意力图
#         spatial_att = self.attention(pool_features)
#         return x * spatial_att

# class DynamicAttentionFusion(nn.Module):  #双重注意力机制+多尺度特征提取
#     def __init__(self, in_channels):
#         super(DynamicAttentionFusion, self).__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 16, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 16, in_channels, 1),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, in_channels // 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 16, 1, kernel_size=5, padding=2),
#             nn.Sigmoid()
#         )
#         self.multi_scale = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 2, in_channels, kernel_size=5, padding=2)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 多尺度特征提取
#         x_ms = self.multi_scale(x)
#         # 通道注意力
#         channel_att = self.channel_attention(x_ms)
#         # channel_att = self.channel_attention(x) + 1e-4 * torch.norm(channel_att, p=2)# 在注意力机制的输出上添加 L2 正则化，避免过度关注噪声区域
#         x = x * channel_att
#         # 空间注意力
#         spatial_att = self.spatial_attention(torch.cat([torch.max(x, dim=1, keepdim=True)[0],
#                                                         torch.mean(x, dim=1, keepdim=True)], dim=1))
#         return x * spatial_att




# class FrequencyEnhancement(nn.Module):
#     def __init__(self, in_channels):
#         super(FrequencyEnhancement, self).__init__()
#         self.low_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.high_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         low_freq = self.low_pass_filter(x)
#         high_freq = x - low_freq
#         enhanced_low_freq = self.relu(low_freq)
#         enhanced_high_freq = self.relu(high_freq)
#         out = enhanced_low_freq + enhanced_high_freq
#         return out






# class FrequencyEnhancement(nn.Module):   #改进的高低频分离处理
#     def __init__(self, in_channels):
#         super(FrequencyEnhancement, self).__init__()
#         self.low_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)  # 深度卷积
#         self.high_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.denoise = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         low_freq = self.low_pass_filter(x)
#         high_freq = x - low_freq
#         high_freq = self.denoise(high_freq)  # 降噪高频部分
#         # 频域降噪（可选）
#         x_freq = torch.fft.fft2(high_freq)
#         h, w = x_freq.shape[-2:]
#         mask = torch.ones_like(x_freq)
#         mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0  # 屏蔽高频
#         x_freq = x_freq * mask
#         high_freq = torch.fft.ifft2(x_freq).real
#         return self.relu(low_freq + high_freq)

class FrequencyEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.high_pass_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.denoise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.threshold = nn.Parameter(torch.tensor(0.1))                                          # 可学习阈值
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        low_freq = self.low_pass_filter(x)
        high_freq = x - low_freq
        high_freq = self.denoise(high_freq)
        x_freq = torch.fft.fft2(high_freq)
        mask = torch.abs(x_freq) > self.threshold
        x_freq = x_freq * mask
        high_freq = torch.fft.ifft2(x_freq).real
        return self.relu(low_freq + high_freq)


#降噪模块
# class NonLocalBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(NonLocalBlock, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = in_channels // 2
#         self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
#         theta_x = self.theta(x).view(b, self.inter_channels, -1)
#         phi_x = self.phi(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
#         f = torch.matmul(theta_x, phi_x)
#         f_div_C = torch.nn.functional.softmax(f, dim=-1)
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous().view(b, self.inter_channels, h, w)
#         return x + self.out_conv(y)



# class NonLocalBlock(nn.Module):
#     def __init__(self, in_channels, inter_channels=None):
#         super(NonLocalBlock, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels or in_channels // 2
#         self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
#         self.bn = nn.BatchNorm2d(in_channels)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 计算全局相似性
#         g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [b, h*w, inter_channels]
#         theta_x = self.theta(x).view(b, self.inter_channels, -1)  # [b, inter_channels, h*w]
#         phi_x = self.phi(x).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [b, h*w, inter_channels]
#         f = torch.matmul(theta_x, phi_x)  # [b, h*w, h*w]
#         f_div_C = torch.nn.functional.softmax(f, dim=-1)  # 归一化相似性
#         y = torch.matmul(f_div_C, g_x)  # 加权求和
#         y = y.permute(0, 2, 1).contiguous().view(b, self.inter_channels, h, w)
#         y = self.out_conv(y)
#         return self.bn(x + y)  # 残差连接
#
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, downsample_factor=0.5):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 4
        self.downsample_factor = downsample_factor
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        # 下采样
        if self.downsample_factor < 1.0:
            x_down = nn.functional.interpolate(x, scale_factor=self.downsample_factor, mode='bilinear')
        else:
            x_down = x
        dh, dw = x_down.size(2), x_down.size(3)
        # 计算特征
        g_x = self.g(x_down).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [b, dh*dw, inter_channels]
        theta_x = self.theta(x_down).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [b, dh*dw, inter_channels]
        phi_x = self.phi(x_down).view(b, self.inter_channels, -1)  # [b, inter_channels, dh*dw]
        # 计算相似性
        f = torch.matmul(theta_x, phi_x)  # [b, dh*dw, dh*dw]
        f_div_C = torch.nn.functional.softmax(f, dim=-1)  # [b, dh*dw, dh*dw]
        # 加权求和
        y = torch.matmul(f_div_C, g_x)  # [b, dh*dw, inter_channels]
        y = y.permute(0, 2, 1).contiguous().view(b, self.inter_channels, dh, dw)  # [b, inter_channels, dh, dw]
        # 上采样
        if self.downsample_factor < 1.0:
            y = nn.functional.interpolate(y, size=(h, w), mode='bilinear')
        y = self.out_conv(y)
        return self.bn(x + y)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv_dilate = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv_dilate(x)
        return self.relu(self.bn(torch.cat([x1, x2, x3, x4], dim=1)))

