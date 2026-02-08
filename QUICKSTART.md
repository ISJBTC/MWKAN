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

### Step 3: Prepare Your Data

Create the following folder structure:

```
MWKAN/
├── morlet_wavelet_kan.py
└── Balanced/
    ├── benign/       ← Put benign images here
    └── malignant/    ← Put malignant images here
```

### Step 4: Run Training

```bash
python morlet_wavelet_kan.py
```

### Step 5: Check Results

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

## ❓ Troubleshooting

**Problem**: `FileNotFoundError: './Balanced/benign'`  
**Solution**: Create `Balanced/benign/` and `Balanced/malignant/` folders with images

**Problem**: `CUDA out of memory`  
**Solution**: Reduce batch_size from 64 to 32 or 16

**Problem**: `ModuleNotFoundError: No module named 'torch'`  
**Solution**: Install PyTorch: `pip install torch torchvision`

## 📞 Need Help?

- Check the full [README.md](README.md)
- Open an issue on [GitHub](https://github.com/ISJBTC/MWKAN/issues)
- Review code comments in `morlet_wavelet_kan.py`

---

**Ready to classify breast cancer histopathology images with Morlet Wavelet-KAN!** 🎯
