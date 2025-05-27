Infrared Image Enhancement with U-Net
This project implements a U-Net-based deep learning model for enhancing low-quality infrared images, incorporating advanced modules such as dynamic attention fusion, frequency enhancement, and non-local blocks. The model is designed to improve image clarity and quality, targeting applications in computer vision tasks like surveillance and thermal imaging.
Features

U-Net Architecture: A convolutional neural network with encoder-decoder structure for pixel-wise image enhancement.
Dynamic Attention Fusion: Combines channel and spatial attention mechanisms to focus on important image features.
Frequency Enhancement: Separates and enhances low and high-frequency components of images with denoising capabilities.
Non-Local Block: Captures long-range dependencies to improve texture and detail preservation.
Multi-Scale Convolution: Extracts features at different scales for robust feature representation.
Custom Loss Function: Uses a hybrid MSE-SSIM loss to optimize both pixel-level accuracy and structural similarity.
Evaluation Metrics: Computes PSNR, SSIM, and MAE to assess enhancement quality.

Installation
Prerequisites

Python 3.8+
PyTorch 1.9+ (with CUDA support for GPU acceleration)
OpenCV, NumPy, Pillow, PyYAML, scikit-image, torchmetrics
Optional: TensorBoard for training visualization

Setup

Clone the repository:
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


Install dependencies:
pip install torch torchvision opencv-python numpy pillow pyyaml scikit-image torchmetrics


Prepare the dataset:

Place low-quality infrared images (16-bit TIFF) in data/train/low_500/ and data/test/low/.
Place high-quality target images (8-bit JPG) in data/train/high_500/ and data/test/high/.
Update configs/train_config.yaml with appropriate paths if needed.



Usage
Training
Train the U-Net model using the provided script:
python train.py


Configures training parameters (e.g., batch size, learning rate) in configs/train_config.yaml.
Saves the best model checkpoint to models/checkpoints/C_best_model_unet29.pth.

Inference
Enhance a single image or a batch of images:
python inference.py


Single Image:
Input: Path to a low-quality TIFF image.
Output: Enhanced image saved as JPG in output/.
Example:python inference.py --input_path data/test/low/video-57kWWRyeqqHs3Byei-frame-000816-b6tuLjNco8MfoBs3d.tiff --target_dir data/test/high --output_path output/imp001.jpg --output_format jpg




Batch Processing:
Processes all images in the input directory and saves results to the output directory.
Example:python inference.py --input_dir data/train/low_60/ --target_dir data/train/high_60 --output_dir output/C_best_model_unet29/ --output_format jpg





Evaluation
Evaluate the model on a validation set:
python evaluate.py


Computes PSNR, SSIM, and MAE metrics.
Saves enhanced images to output/001/ and comparison samples to evaluation_samples/.
Example output:评估结果 (models/checkpoints/C_best_model_unet03.pth):
PSNR均值: 25.43 dB
SSIM均值: 0.8732
MAE均值: 0.0421



Project Structure
├── configs/
│   └── train_config.yaml      # Configuration file for training
├── data/
│   ├── train/
│   │   ├── low_500/          # Low-quality training images (TIFF)
│   │   └── high_500/         # High-quality target images (JPG)
│   ├── val/
│   │   ├── low_50/           # Low-quality validation images
│   │   └── high_50/          # High-quality validation targets
│   └── test/
│       ├── low/              # Low-quality test images
│       └── high/             # High-quality test targets
├── models/
│   ├── checkpoints/           # Model checkpoints
│   ├── base_unet.py          # Basic U-Net implementation
│   └── components.py         # Custom modules (attention, frequency, non-local)
├── output/                    # Enhanced images and comparison samples
├── evaluation_samples/        # Visualization of input, enhanced, and target images
├── train.py                   # Training script
├── inference.py               # Inference script for single/batch processing
├── evaluate.py                # Evaluation script for metrics
└── README.md                  # This file

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.

Please include clear commit messages and update documentation if needed. For major changes, open an issue first to discuss.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Notes

The project is in an early stage, and some modules (e.g., advanced attention mechanisms) are commented out for experimentation. Contributions to optimize these are appreciated.
Current model performance may vary with different datasets. Ensure input images are properly preprocessed (16-bit TIFF) to avoid issues like white artifacts.
For questions or issues, please open an issue on GitHub.

