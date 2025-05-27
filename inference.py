import os
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from aaainfrared_image_fusion.models.base_unet import UNet
from aaainfrared_image_fusion.models.improved_unet import ImprovedUNet

class InfraredEnhancer:
    def __init__(self, model_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model = UNet(n_channels=1, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def preprocess(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件 {input_path} 不存在")
        raw_img = cv2.imread(input_path, -1).astype(np.float32)
        processed_img = cv2.normalize(raw_img, None, 0, 1, cv2.NORM_MINMAX)
        input_tensor = self.transform(Image.fromarray((processed_img * 255).astype(np.uint8)))
        return input_tensor.unsqueeze(0).to(self.device)

    def postprocess(self, output_tensor):
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_img = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        return output_img

    def enhance_single(self, input_path, target_dir, output_path=None, show_comparison=True, output_format="tiff"):
        input_tensor = self.preprocess(input_path)
        import time
        start_time = time.time()
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time:.4f} 秒")

        enhanced_img = self.postprocess(output_tensor)
        raw_img = cv2.imread(input_path, -1).astype(np.float32)
        raw_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        target_path = os.path.join(target_dir, base_name + ".jpg")
        if not os.path.exists(target_path):
            print(f"警告: 目标文件 {target_path} 不存在，跳过对比图显示")
            target_img = np.zeros_like(raw_img)
        else:
            target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if show_comparison and output_path:
            comparison = np.hstack([raw_img, enhanced_img, target_img])
            comparison_path = os.path.join(os.path.dirname(output_path), "comparison.jpg")
            cv2.imwrite(comparison_path, comparison)
            print(f"对比图已保存至 {comparison_path}")

        if output_path:
            # 动态调整输出路径扩展名
            if output_format.lower() == "jpg":
                output_path = os.path.splitext(output_path)[0] + ".jpg"
            cv2.imwrite(output_path, enhanced_img)

        return enhanced_img

    def enhance_batch(self, input_dir, target_dir, output_dir, output_format="tiff"):
        os.makedirs(output_dir, exist_ok=True)
        filenames = sorted(os.listdir(input_dir))
        for filename in filenames:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{output_format}")
            self.enhance_single(input_path, target_dir, output_path, show_comparison=False, output_format=output_format)
        print(f"批量处理完成，共处理{len(filenames)}张图像")

if __name__ == "__main__":
    enhancer = InfraredEnhancer(
        model_path="models/checkpoints/C_best_model_unet29.pth",
        config_path="./configs/train_config.yaml"
    )

    # 单图测试模式
    enhancer.enhance_single(
        input_path="./data/test/low/video-57kWWRyeqqHs3Byei-frame-000816-b6tuLjNco8MfoBs3d.tiff",
        target_dir="./data/test/high",
        output_path="./output/imp001.jpg",
        show_comparison=True,
        output_format="jpg"  # 指定输出为 JPG
    )
    enhancer.enhance_batch(
        input_dir="./data/train/low_60/",
        target_dir="./data/train/high_60",
        output_dir="./output/C_best_model_unet29/",
        output_format="jpg"  # 指定批量输出为 JPG
    )