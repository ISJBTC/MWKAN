# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone the Repository

```bash
git clone https://github.com/ISJBTC/MWKAN.git
cd MWKAN
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

**Dataset**: [Kaggle - Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

1. Visit: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
2. Download the dataset (requires free Kaggle account)
3. Extract the downloaded archive
4. The dataset contains 277,524 patches (50×50 pixels)
   - IDC Negative (benign): 198,738 patches
   - IDC Positive (malignant): 78,786 patches

### Step 4: Prepare Your Data

Organize images into this structure:

```
MWKAN/
├── morlet_wavelet_kan.py
└── Balanced/
    ├── benign/       ← Put IDC-negative images here
    └── malignant/    ← Put IDC-positive images here
```

### Step 5: Run Training

```bash
python morlet_wavelet_kan.py
```

### Step 6: Check Results

After training completes (~30-60 minutes), you'll have:
- `ImprovedKAN_best_model_morlet.pt` - Trained model
- `morlet_*.png` - Training curves
- `morlet_confusion_matrix.png` - Confusion matrix
- Console output with test accuracy

## 📊 Expected Output

```
TRAINING STARTED
================================================================================
Epoch [1/50], Train Loss: 0.XXXX, Train Accuracy: 0.XXXX, Val Loss: 0.XXXX...
...
Best Validation Accuracy: 0.XXXX at Epoch XX

TEST RESULTS
================================================================================
Accuracy:  0.XXXX (XX.XX%)
Precision: 0.XXXX
Recall:    0.XXXX
F1 Score:  0.XXXX
```

## 🔧 Customization

To change training parameters, edit these lines in `morlet_wavelet_kan.py`:

```python
# Around line 175-185
batch_size = 64          # Change batch size
num_epochs = 50          # Change number of epochs
learning_rate = 0.0001   # Change learning rate
```

## 📚 Dataset Citation

If you use this code, please cite the dataset:

```bibtex
@article{janowczyk2016deep,
  title={Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases},
  author={Janowczyk, Andrew and Madabhushi, Anant},
  journal={Journal of Pathology Informatics},
  volume={7},
  pages={29},
  year={2016},
  doi={10.4103/2153-3539.186902}
}
```

## ❓ Troubleshooting

**Problem**: `FileNotFoundError: './Balanced/benign'`  
**Solution**: Create `Balanced/benign/` and `Balanced/malignant/` folders with images from Kaggle dataset

**Problem**: `CUDA out of memory`  
**Solution**: Reduce batch_size from 64 to 32 or 16

**Problem**: `ModuleNotFoundError: No module named 'torch'`  
**Solution**: Install PyTorch: `pip install torch torchvision`

**Problem**: Need Kaggle dataset access  
**Solution**: Create free account at kaggle.com and download dataset

## 📞 Need Help?

- Check the full [README.md](README.md)
- Open an issue on [GitHub](https://github.com/ISJBTC/MWKAN/issues)
- Review code comments in `morlet_wavelet_kan.py`
- Dataset page: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

---

**Ready to classify breast cancer histopathology images with Morlet Wavelet-KAN!** 🎯

**Dataset**: 277,524 histopathology patches from Kaggle
