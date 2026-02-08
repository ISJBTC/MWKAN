import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from torchsummary import summary  # Importing summary from torchsummary

# Custom Linear Layer with Wavelet Transformation
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0)
        scale_expanded = self.scale.unsqueeze(0)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            
            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            wavelet = sinc * window
        else:
            raise ValueError("Unsupported wavelet type")

        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output
        return self.bn(combined_output)

# Improved Neural Network with Custom Layers
class ImprovedKAN(nn.Module):
    def __init__(self, input_size, num_classes=2, wavelet_type='morlet'):
        super(ImprovedKAN, self).__init__()
        self.layers = nn.ModuleList()
        layers_hidden = [input_size, 128, 128, 64, 32]  # Increased hidden layers
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))
        self.output_layer = nn.Linear(layers_hidden[-1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Function to Get List of Files
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isfile(fullPath):
            allFiles.append(fullPath)
    return allFiles

# Validate Images
def validate_images(image_paths):
    valid_images = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img.verify()
                valid_images.append(img_path)
        except (UnidentifiedImageError, IOError):
            print(f"Invalid image file: {img_path}")
    return valid_images

# Load Image Paths and Labels
benign_images = validate_images(getListOfFiles('./Balanced/benign'))
malignant_images = validate_images(getListOfFiles('./Balanced/malignant'))

data = pd.DataFrame(index=np.arange(0, len(benign_images) + len(malignant_images)), columns=["image", "target"])
k = 0
for image_path in benign_images:
    data.loc[k, "image"] = image_path
    data.loc[k, "target"] = 0
    k += 1
for image_path in malignant_images:
    data.loc[k, "image"] = image_path
    data.loc[k, "target"] = 1
    k += 1

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image"]
        label = self.dataframe.iloc[idx]["target"]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# Split Data into Train, Validation, and Test Sets
train_df, val_df, test_df = np.split(data.sample(frac=1, random_state=42), [int(.6*len(data)), int(.8*len(data))])

# Define wavelet types
wavelet_types = ['morlet']

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
train_dataset = MyDataset(dataframe=train_df, transform=transform)
val_dataset = MyDataset(dataframe=val_df, transform=transform)
test_dataset = MyDataset(dataframe=test_df, transform=transform)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Input Size
def get_input_size(dataset):
    sample, _ = dataset[0]
    return sample.numel()

input_size = get_input_size(train_dataset)

# Calculate Metrics
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy

# Plot Metrics
def plot_metrics(epochs, train_metrics, val_metrics, test_metric, metric_name, wavelet_type):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, label=f'Validation {metric_name}')
    plt.axhline(y=test_metric, color='r', linestyle='--', label=f'Test {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs for {wavelet_type} Wavelet')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{wavelet_type}_{metric_name.lower()}.png')
    plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, wavelet_type, title='Confusion Matrix', normalize=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{wavelet_type}_confusion_matrix.png')
    plt.show()

# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, wavelet_type, num_epochs=50, verbose=True):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_labels = []
        all_train_preds = []
        for inputs, labels in tqdm(train_loader, disable=not verbose):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_precision, train_recall, train_f1, _ = calculate_metrics(all_train_labels, all_train_preds)

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()

                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_precision, val_recall, val_f1, _ = calculate_metrics(all_val_labels, all_val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        epochs.append(epoch + 1)

        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            print(f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}')
            print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Save the model with best validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'ImprovedKAN_best_model_{wavelet_type}.pt')

    print(f'Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}')

    # Plotting loss and accuracy curves
    plot_metrics(epochs, train_losses, val_losses, best_val_acc, "Loss", wavelet_type)
    plot_metrics(epochs, train_accs, val_accs, best_val_acc, "Accuracy", wavelet_type)
    plot_metrics(epochs, train_precisions, val_precisions, best_val_acc, "Precision", wavelet_type)
    plot_metrics(epochs, train_recalls, val_recalls, best_val_acc, "Recall", wavelet_type)
    plot_metrics(epochs, train_f1s, val_f1s, best_val_acc, "F1 Score", wavelet_type)

    return train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s

# Initialize Model, Criterion, and Optimizer
# FIXED: Changed from 'meyer' to 'morlet' to match the wavelet_type variable below
model = ImprovedKAN(input_size=input_size, num_classes=2, wavelet_type='morlet').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Print Model Summary
summary(model, (1, 128, 128))  # Assuming grayscale images, hence 1 channel

# Train the Model
wavelet_type = 'morlet'
train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s = train_model(model, criterion, optimizer, trainloader, valloader, wavelet_type, num_epochs=50)

# Evaluate on Test Set
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    precision, recall, f1, accuracy = calculate_metrics(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=['Benign', 'Malignant'], wavelet_type='morlet', title='Confusion Matrix')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

# Load the best model
model.load_state_dict(torch.load(f'ImprovedKAN_best_model_morlet.pt'))

# Evaluate the model on the test set
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, testloader)
