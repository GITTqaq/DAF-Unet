import os
import cv2
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from datetime import datetime
from aaainfrared_image_fusion.models.base_unet import UNet
from aaainfrared_image_fusion.models.multi_scale_unet import MultiScaleUNet
from aaainfrared_image_fusion.models.DAF_Unet import DAF_UNet
from aaainfrared_image_fusion.losses.custom_loss import HybridLoss
from aaainfrared_image_fusion.losses.MSE_SSIM_Loss import MSE_SSIM_Loss
from aaainfrared_image_fusion.models.improved_unet import ImprovedUNet
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("runs/experiment")

# 配置参数
with open("./configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

import sys


# print(sys.path)# 自定义数据集类
class InfraredDataset(Dataset):
    def __init__(self, img_dir, target_dir, transform=None):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __getitem__(self, idx):
        # 加载16位TIFF输入
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        input_img = cv2.imread(img_path, -1)  # 使用cv2读取16位图像
        input_img = input_img.astype(np.float32)  # 转换为32位浮点

        # # 应用 CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # input_img = clahe.apply(input_img.astype(np.uint16)).astype(np.float32)
        # input_img = cv2.normalize(input_img, None, 0, 1, cv2.NORM_MINMAX)

        # 预处理步骤
        # 1. 高斯模糊
        # input_img = cv2.GaussianBlur(input_img, (5, 5), 0)

        # 2. 自适应 CLAHE
        # std = np.std(input_img)
        # clip_limit = 2.0 #if std > 0.1 else 3.0  # 高噪声图像用低 clip_limit
        # tile_size = (4, 4) #if std > 0.1 else (16, 16)  # 高噪声用小 tile
        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        # input_img = clahe.apply(input_img.astype(np.uint16)).astype(np.float32)
        #
        # # 3. 对数变换
        # input_img = np.log1p(input_img) / np.log1p(input_img.max())
        #
        #


        # 加载8位JPG目标
        target_path = os.path.join(self.target_dir,
                                   os.path.splitext(self.img_names[idx])[0] + ".jpg")
        target_img = Image.open(target_path).convert('L')  # 单通道灰度

        # 动态范围压缩（关键步骤）
        input_img = cv2.normalize(input_img, None, 0, 1, cv2.NORM_MINMAX)  # 归一化到[0,1]

        # 转换为Tensor
        if self.transform:
            # 确保输入是正确的张量维度
            input_tensor = self.transform(Image.fromarray((input_img * 255).astype(np.uint8)))
            target_tensor = self.transform(target_img)
        else:
            # 当没有transform时的默认处理
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).float()  # 添加通道维度
            target_tensor = torch.from_numpy(np.array(target_img)).unsqueeze(0).float() / 255.0

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.img_names)  # 返回数据集的样本数量


# 数据转换增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# 数据转换增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强增强
# 初始化数据加载器
train_dataset = InfraredDataset(
    img_dir="data/train/low_500",
    target_dir="data/train/high_500",
    # transform=train_transform
    transform=None
)
val_dataset = InfraredDataset(
    img_dir="data/val/low_50",
    target_dir="data/val/high_50",
    # transform=val_transform
    transform=None
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2
)

# 模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=1).to(device)
# model = MultiScaleUNet(n_channels=1, n_classes=1).to(device)  # 假设UNet已正确导入
# model = ImprovedUNet(n_channels=1, n_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] // 10, eta_min=1e-6)
criterion = MSE_SSIM_Loss(mse_weight=0.6, ssim_weight=0.4).to(device)
# criterion = nn.MSELoss()


# loss_config = {
#     'loss_weights': {
#         'mse': 0.5,
#         'ssim': 0.3,
#         'vgg': 0.2
#     }
# }
# criterion = HybridLoss(loss_config)

# 训练循环
def train_model(epochs):
    best_val_loss = np.inf
    best_model_path = "models/checkpoints/C_best_model_unet29.pth"
    patience_counter = 0
    patience = 6

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            print(f"训练阶段 - 当前批次损失: {loss.item()}")
            # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                #归一化的部分，输出偏白 1111111111111111111111111111111
                # outputs = torch.clamp(outputs, 0, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                print(f"验证阶段 - 当前批次损失: {loss.item()}")  #

        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型: {best_model_path} (验证损失: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"在第 {epoch + 1} 个 epoch 早停")
                break

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f"输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        # writer.add_scalar("Loss/Train", train_loss, epoch)
        # writer.add_scalar("Loss/Val", val_loss, epoch)
        # writer.add_images("Samples", outputs[:4], epoch)


if __name__ == "__main__":
    train_model(epochs=config['epochs'])
