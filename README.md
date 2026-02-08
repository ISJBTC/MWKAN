# Morlet Wavelet-KAN for Breast Cancer Histopathology Classification

[![DOI](https://zenodo.org/badge/1152859414.svg)](https://doi.org/10.5281/zenodo.18525841)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of Morlet Wavelet-enhanced Kolmogorov-Arnold Network (KAN) for binary classification of breast cancer histopathology images.

## Overview

This repository contains the complete implementation of a novel deep learning architecture that integrates Morlet wavelet transformations with Kolmogorov-Arnold Network layers for accurate classification of breast cancer histopathology images into benign and malignant categories.

### Key Features

- **Morlet Wavelet Integration**: Custom wavelet transformation layers using Morlet wavelet (ω₀ = 5.0)
- **KAN Architecture**: Multi-layer network with learnable wavelet parameters
- **Binary Classification**: Benign vs Malignant histopathology image classification
- **Comprehensive Evaluation**: Full metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Reproducible Results**: Fixed random seed (42) for consistent results

## Architecture

The network implements a custom `KANLinear` layer that combines:
- Wavelet transformation with learnable scale and translation parameters
- Base linear transformation
- Batch normalization

**Network Structure:**
```
Input (128×128 grayscale images) → Flatten → KAN Layers → Output (2 classes)
Hidden layers: [128, 128, 64, 32]
```

### Morlet Wavelet Formula

The Morlet wavelet transformation used in each KAN layer:

```
ψ(x) = cos(ω₀ · x) · exp(-x²/2)
```

where ω₀ = 5.0 (standard Morlet parameter)

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
Pillow>=8.0.0
tqdm>=4.50.0
torchsummary>=1.5.0
openpyxl>=3.0.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your histopathology images as follows:

```
project_root/
├── morlet_wavelet_kan.py
└── Balanced/
    ├── benign/
    │   ├── image001.png
    │   ├── image002.png
    │   └── ...
    └── malignant/
        ├── image001.png
        ├── image002.png
        └── ...
```

**Supported image formats**: PNG, JPG, JPEG, BMP

## Usage

### Training

```bash
python morlet_wavelet_kan.py
```

The script will:
1. Load and validate images from `./Balanced/benign/` and `./Balanced/malignant/`
2. Split data: 60% training, 20% validation, 20% testing
3. Train the model for 50 epochs
4. Save the best model based on validation accuracy
5. Evaluate on the test set
6. Generate performance plots and confusion matrix

### Training Parameters

Default configuration:
- **Image size**: 128×128 pixels
- **Batch size**: 64
- **Learning rate**: 0.0001
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **Epochs**: 50
- **Wavelet type**: Morlet

To modify parameters, edit the configuration section in the code (lines ~140-160).

## Outputs

After training, the following files are generated:

### Model Files
- `ImprovedKAN_best_model_morlet.pt` - Best trained model weights

### Visualizations
- `morlet_loss.png` - Training and validation loss curves
- `morlet_accuracy.png` - Training and validation accuracy curves
- `morlet_precision.png` - Precision over epochs
- `morlet_recall.png` - Recall over epochs
- `morlet_f1_score.png` - F1 score over epochs
- `morlet_confusion_matrix.png` - Confusion matrix on test set

### Performance Metrics

The model outputs comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Model Architecture Details

### KANLinear Layer

Each KAN layer implements:

1. **Wavelet Transformation**:
   - Learnable scale parameters: `scale[out_features, in_features]`
   - Learnable translation parameters: `translation[out_features, in_features]`
   - Wavelet-specific weights: `wavelet_weights[out_features, in_features]`

2. **Base Transformation**:
   - Linear transformation: `weight1[out_features, in_features]`

3. **Batch Normalization**:
   - Applied to combined output

### Network Layers

```python
Layer 1: 16384 → 128 (KANLinear + BatchNorm)
Layer 2: 128 → 128 (KANLinear + BatchNorm)
Layer 3: 128 → 64 (KANLinear + BatchNorm)
Layer 4: 64 → 32 (KANLinear + BatchNorm)
Output: 32 → 2 (Linear)
```

## Data Preprocessing

Images are preprocessed with:
1. Conversion to grayscale
2. Resize to 128×128 pixels
3. Tensor conversion
4. Normalization: mean=0.5, std=0.5

## Reproducibility

For reproducible results:
- Random seed: 42 (set in data splitting)
- Fixed train/validation/test split ratios
- Deterministic data loading

## Citation

If you use this code in your research, please cite:

```bibtex
@software{jamadar2025mwkan,
  author = {Jamadar, Irshad},
  title = {Morlet Wavelet-KAN for Breast Cancer Histopathology Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ISJBTC/MWKAN},
  doi = {10.5281/zenodo.18525841}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MIT Art, Design and Technology University, Pune
- Supervisor: Prof. Dr. Krishna Kumar
- Department of Applied Science and Humanities

## Contact

**Author**: Irshad Jamadar  
**Institution**: MIT Art, Design and Technology University, Pune  
**Registration**: MITU22PHMT0002  
**Email**: irshadjamadar@mituniversity.edu

## Technical Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/ISJBTC/MWKAN/issues)
- Check existing documentation
- Review the code comments

## Version History

- **v1.0.0** (2025-02-08): Initial release
  - Complete Morlet Wavelet-KAN implementation
  - Binary classification for breast cancer histopathology
  - Comprehensive evaluation metrics
  - Training visualization tools

---

**Note**: This implementation is part of PhD research on intelligent approaches for breast cancer prediction using mathematical techniques.
