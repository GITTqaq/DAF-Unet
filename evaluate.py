import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from aaainfrared_image_fusion.models.base_unet import UNet
from train import InfraredDataset, val_transform
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_psnr(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate_model(config_path, model_weights, save_samples=True, save_enhanced_images=True):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if 'validation' not in config:
        # print(f"警告: 配置文件 {config_path} 中缺少 'validation' 部分，使用默认路径")
        val_config = {
            # 'low_dir': "./data/test/low",
            # 'high_dir': "./data/test/high"
            'low_dir': "./data/train/low_60",
            'high_dir': "./data/train/high_60"

        }
    else:
        val_config = config['validation']

    # 初始化数据集
    try:
        val_dataset = InfraredDataset(
            img_dir=val_config['low_dir'],
            target_dir=val_config['high_dir'],
            transform=None
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    except KeyError as e:
        print(f"错误: 在验证数据集配置中缺少必要的键: {e}")
        return

    # 加载模型
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_weights, map_location=DEVICE))
    model.eval()

    # 指标容器
    psnr_values = []
    ssim_values = []
    mae_values = []

    if save_enhanced_images:
        enhanced_images_dir = "output/001/"
        os.makedirs(enhanced_images_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="验证进度")):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            # 输出后处理：与 inference.py 一致
            outputs = torch.clamp(outputs, 0, 1)  # 确保输出在 [0, 1]

            # 指标计算
            psnr = calculate_psnr(outputs, targets).item()
            ssim_val = ssim_metric(outputs, targets, data_range=1.0).item()
            mae = torch.mean(torch.abs(outputs - targets)).item()

            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
            mae_values.append(mae)

            # 样本可视化保存
            if save_samples and idx % 50 == 0:
                sample_dir = "evaluation_samples"
                os.makedirs(sample_dir, exist_ok=True)

                # 反归一化处理：与 inference.py 一致
                input_img = inputs[0].cpu().numpy().squeeze() * 255  # 输入已在 [0, 1]
                output_img = outputs[0].cpu().numpy().squeeze() * 255  # 输出已在 [0, 1]
                target_img = targets[0].cpu().numpy().squeeze() * 255  # 目标已在 [0, 1]

                # 拼接对比图
                comparison = np.hstack([input_img.astype(np.uint8),
                                        output_img.astype(np.uint8),
                                        target_img.astype(np.uint8)])
                cv2.imwrite(f"{sample_dir}/comparison_{idx:04d}.jpg", comparison)

            # 保存增强后的图像
            if save_enhanced_images:
                enhanced_img = outputs[0].cpu().numpy().squeeze() * 255
                enhanced_img = enhanced_img.astype(np.uint8)
                original_filename = val_dataset.img_names[idx]
                base_name = os.path.splitext(original_filename)[0]
                new_filename = f"{base_name}.jpg"
                cv2.imwrite(os.path.join(enhanced_images_dir, new_filename), enhanced_img)

    # 汇总统计结果
    print(f"\n评估结果 ({model_weights}):")
    print(f"PSNR均值: {np.mean(psnr_values):.2f} dB")
    print(f"SSIM均值: {np.mean(ssim_values):.4f}")
    print(f"MAE均值: {np.mean(mae_values):.4f}")

    return {
        "psnr": np.mean(psnr_values),
        "ssim": np.mean(ssim_values),
        "mae": np.mean(mae_values)
    }

if __name__ == "__main__":
    evaluate_model(
        config_path="./configs/train_config.yaml",
        model_weights="models/checkpoints/C_best_model_unet03.pth"  # 与 inference.py 一致
    )