# Morlet Wavelet-KAN for Breast Cancer Histopathology Classification

[![DOI](https://zenodo.org/badge/1152859414.svg)](https://doi.org/10.5281/zenodo.18525841)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

Implementation of Morlet Wavelet-enhanced Kolmogorov-Arnold Network (KAN) for binary classification of breast cancer histopathology images.

## Overview

This repository contains the complete implementation of a novel deep learning architecture that integrates Morlet wavelet transformations with Kolmogorov-Arnold Network layers for accurate classification of breast cancer histopathology images into benign and malignant categories.

### Key Features

- **Morlet Wavelet Integration**: Custom wavelet transformation layers using Morlet wavelet (ω₀ = 5.0)
- **KAN Architecture**: Multi-layer network with learnable wavelet parameters
- **Binary Classification**: Benign vs Malignant histopathology image classification
- **Comprehensive Evaluation**: Full metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Reproducible Results**: Fixed random seed (42) for consistent results

## Dataset

This implementation uses the **Breast Histopathology Images** dataset from Kaggle:

**Source**: [Kaggle - Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

**Dataset Details**:
- **Total Images**: 277,524 patches (50×50 pixels each)
- **Classes**: 
  - IDC Negative (benign): 198,738 patches
  - IDC Positive (malignant): 78,786 patches
- **Image Format**: PNG
- **Original Resolution**: Extracted from 162 whole mount slide images
- **Staining**: H&E stained breast tissue samples

**Citation**:
```bibtex
@article{janowczyk2016deep,
  title={Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases},
  author={Janowczyk, Andrew and Madabhushi, Anant},
  journal={Journal of Pathology Informatics},
  volume={7},
  pages={29},
  year={2016},
  publisher={Wolters Kluwer--Medknow Publications},
  doi={10.4103/2153-3539.186902}
}
```

**Note**: The dataset patches are originally 50×50 pixels. Our implementation resizes them to 128×128 pixels for processing.

### How to Download Dataset

1. **Visit Kaggle**: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
2. **Download** the dataset (requires free Kaggle account)
3. **Extract** the downloaded archive
4. **Organize** images into the required structure (see Dataset Preparation section)

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

## Dataset Preparation

After downloading from Kaggle, organize your histopathology images as follows:

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

**Steps to prepare:**

1. **Download** dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
2. **Extract** all images from the downloaded archive
3. **Separate** IDC-negative images into `Balanced/benign/`
4. **Separate** IDC-positive images into `Balanced/malignant/`
5. **Balance** the dataset if desired (optional - subsample majority class)

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
- **Image size**: 128×128 pixels (resized from original 50×50)
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
2. Resize from 50×50 to 128×128 pixels
3. Tensor conversion
4. Normalization: mean=0.5, std=0.5

## Reproducibility

For reproducible results:
- Random seed: 42 (set in data splitting)
- Fixed train/validation/test split ratios (60/20/20)
- Deterministic data loading
- Fixed dataset source (Kaggle)

## Citation

If you use this code in your research, please cite:

**This Software:**
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

**Dataset:**
```bibtex
@article{janowczyk2016deep,
  title={Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases},
  author={Janowczyk, Andrew and Madabhushi, Anant},
  journal={Journal of Pathology Informatics},
  volume={7},
  pages={29},
  year={2016},
  publisher={Wolters Kluwer--Medknow Publications},
  doi={10.4103/2153-3539.186902}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The dataset used is subject to its own terms of use. Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) for dataset licensing information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- **Dataset**: Paul Mooney (Kaggle) for curating and sharing the Breast Histopathology Images dataset

## Contact

**Author**: Irshad Jamadar  
**Institution**: MIT Art, Design and Technology University, Pune   
**Email**: jamadarirshad@gmail.com

## Technical Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/ISJBTC/MWKAN/issues)
- Check existing documentation
- Review the code comments

## Version History

- **v1.0.0** (2025-02-08): Initial release
  - Complete Morlet Wavelet-KAN implementation
  - Binary classification for breast cancer histopathology
  - Trained on Kaggle Breast Histopathology Images dataset
  - Comprehensive evaluation metrics
  - Training visualization tools

---

**Note**: This implementation is part of PhD research on intelligent approaches for breast cancer prediction using mathematical techniques.

**Dataset Source**: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
