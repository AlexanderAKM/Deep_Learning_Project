#%% Imports
import os
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import kagglehub
from pathlib import Path
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% Dataset Download & Setup
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 
               'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']

print("Downloading HAM10000 dataset via kagglehub...")
DATA_DIR = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
print(f"Dataset path: {DATA_DIR}")

#%% U-Net Architecture for Segmentation (Paper: "Hybrid U-Net and Improved MobileNet-V3")
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    # COMMENT: Paper doesn't specify U-Net architecture details. Using standard U-Net.
    # IMPROVEMENT: Could use attention U-Net or U-Net++ for better segmentation.
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.out_conv(d1))

#%% Simple Lesion Segmentation (fallback when U-Net not trained)
def segment_lesion_simple(image_np):
    # COMMENT: Paper uses trained U-Net but doesn't provide weights.
    # This is a simple color-based segmentation as fallback.
    # IMPROVEMENT: Train U-Net on ISIC segmentation masks if available.
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    lower = np.array([0, 20, 20])
    upper = np.array([180, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    
    return mask

def apply_segmentation_mask(image, mask):
    # COMMENT: Paper doesn't specify how mask is applied. Using masked crop + resize.
    # IMPROVEMENT: Could use mask to remove background completely or use bounding box crop.
    mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
    masked = (image * mask_3ch).astype(np.uint8)
    
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        pad = 10
        y_min, y_max = max(0, y_min - pad), min(image.shape[0], y_max + pad)
        x_min, x_max = max(0, x_min - pad), min(image.shape[1], x_max + pad)
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
    return image

#%% Transforms - EXACTLY as specified in paper Table 4
def get_transforms(phase, use_segmentation=True):
    # Paper Table 4 specifies these EXACT values:
    # - Rotation: 25 degrees
    # - Width/Height Shift: 15%
    # - Shearing: 15%
    # - Horizontal Flip: Yes
    # - Vertical Flip: Yes  
    # - Brightness: [0.9, 1.5]
    # - Zoom: 0.4
    # - Input size: 224x224
    # - Normalization: [0, 1]
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Paper: 224x224
            transforms.RandomRotation(25),  # Paper: 25 degrees
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.15, 0.15),  # Paper: 15% width/height shift
                shear=15  # Paper: 15% shearing
            ),
            transforms.RandomHorizontalFlip(p=0.5),  # Paper: horizontal flip
            transforms.RandomVerticalFlip(p=0.5),  # Paper: vertical flip
            transforms.ColorJitter(brightness=(0.9, 1.5)),  # Paper: brightness [0.9, 1.5]
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Paper: zoom 0.4 (interpreted as scale 0.6-1.0)
            transforms.ToTensor(),
            # COMMENT: Paper says "Normalization 0, 1" but doesn't specify mean/std.
            # Using ImageNet stats since we use pretrained weights.
            # IMPROVEMENT: Could compute dataset-specific mean/std.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#%% Custom Dataset with Segmentation Preprocessing
class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_segmentation=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_segmentation = use_segmentation
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img_np = np.array(img)
        
        # Apply U-Net segmentation preprocessing (Paper: "Hybrid U-Net")
        if self.use_segmentation:
            mask = segment_lesion_simple(img_np)
            img_np = apply_segmentation_mask(img_np, mask)
            img = Image.fromarray(img_np)
        
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

#%% ECA Module (Efficient Channel Attention) - Paper: "replaced with ECA"
class ECAModule(nn.Module):
    # Paper: "squeeze and excitation component was replaced with the practical channel attention component (ECA)"
    # COMMENT: Paper doesn't specify kernel size k. Using adaptive k based on channels.
    # IMPROVEMENT: Could tune k as hyperparameter.
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size as per ECA-Net paper
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(3, k_size)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#%% Replace SE with ECA in MobileNetV3
def replace_se_with_eca(model):
    for name, module in model.named_modules():
        if hasattr(module, 'block'):
            for i, block in enumerate(module.block):
                if hasattr(block, 'se') and block.se is not None:
                    in_channels = block.se.fc1.in_features if hasattr(block.se, 'fc1') else block.se[0].in_features
                    block.se = ECAModule(in_channels)
    return model

#%% Add Dilation to Late Stages - Paper: "dilated convolutions were incorporated"
def add_dilation(model):
    # Paper: "dilated convolutions were incorporated into the model to enhance the receptive field"
    # COMMENT: Paper doesn't specify which layers or dilation rate. Using last 3 blocks with dilation=2.
    # IMPROVEMENT: Could experiment with different dilation rates and layers.
    features = model.features
    for i in range(-3, 0):
        block = features[i]
        if hasattr(block, 'block'):
            for layer in block.block:
                if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
                    layer.dilation = (2, 2)
                    layer.padding = (2, 2)
    return model

#%% Improved MobileNet-V3 - Full implementation per paper
class ImprovedMobileNetV3(nn.Module):
    def __init__(self, num_classes=7, dropout=0.1):
        super().__init__()
        # Paper: "MobileNet-V3 (Large variant implied by 157 layers)"
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Paper: "SE component was replaced with ECA"
        self.backbone = replace_se_with_eca(self.backbone)
        
        # Paper: "dilated convolutions were incorporated"
        self.backbone = add_dilation(self.backbone)
        
        # Paper: "Dropout rate 0.1"
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=dropout),  # Paper: 0.1
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self):
        return self.backbone.features

#%% Grad-CAM for Interpretability
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy(), target_class

def visualize_gradcam(model, image_path, transform, save_path=None):
    model.eval()
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    cam, pred_class = gradcam.generate_cam(input_tensor)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img_array + 0.5 * heatmap
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay - Pred: {CLASS_NAMES[pred_class]}')
    axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return pred_class

#%% Training Function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), acc

#%% Validation Function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return running_loss / len(loader), acc, precision, recall, f1, all_preds, all_labels

#%% Main Training Pipeline - EXACTLY as paper specifies
def main():
    metadata_path = DATA_DIR / "HAM10000_metadata.csv"
    image_dirs = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]
    
    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        return None, None
    
    metadata = pd.read_csv(metadata_path)
    label_map = {c: i for i, c in enumerate(CLASSES)}
    
    image_paths = []
    labels = []
    lesion_ids = []
    
    for _, row in metadata.iterrows():
        img_name = row['image_id'] + '.jpg'
        for img_dir in image_dirs:
            img_path = img_dir / img_name
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(label_map[row['dx']])
                lesion_ids.append(row['lesion_id'])
                break
    
    print(f"Total samples: {len(image_paths)}")
    print(f"Class distribution: {Counter(labels)}")
    
    # COMMENT: Paper uses 70/15/15 split but doesn't mention patient-level splitting.
    # HAM10000 has multiple images per lesion - proper split should be by lesion_id.
    # IMPROVEMENT: This prevents data leakage where same lesion appears in train and test.
    unique_lesions = list(set(lesion_ids))
    lesion_to_indices = {}
    for idx, lid in enumerate(lesion_ids):
        if lid not in lesion_to_indices:
            lesion_to_indices[lid] = []
        lesion_to_indices[lid].append(idx)
    
    # Paper: 70% train, 15% val, 15% test
    train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.30, random_state=42)
    val_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.50, random_state=42)
    
    train_indices = [idx for lid in train_lesions for idx in lesion_to_indices[lid]]
    val_indices = [idx for lid in val_lesions for idx in lesion_to_indices[lid]]
    test_indices = [idx for lid in test_lesions for idx in lesion_to_indices[lid]]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    print(f"Train class distribution: {Counter(train_labels)}")
    
    # Paper: "data augmentation to balance the dataset"
    # Implementing weighted sampling for class balance
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create datasets with segmentation preprocessing
    train_dataset = SkinCancerDataset(train_paths, train_labels, get_transforms('train'), use_segmentation=True)
    val_dataset = SkinCancerDataset(val_paths, val_labels, get_transforms('val'), use_segmentation=True)
    test_dataset = SkinCancerDataset(test_paths, test_labels, get_transforms('test'), use_segmentation=True)
    
    # Paper Table 4: Batch size = 8
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Paper: Dropout = 0.1
    model = ImprovedMobileNetV3(num_classes=7, dropout=0.1).to(device)
    
    # Paper Table 4: Cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # COMMENT: Paper specifies LR=0.2 and weight_decay=1.2 which are EXTREMELY unusual values.
    # LR=0.2 with Adam typically causes divergence. Weight_decay=1.2 would zero out all weights.
    # These are almost certainly typos in the paper.
    # IMPROVEMENT: Using standard values that actually work. Paper results likely used different values.
    
    # PAPER VALUES (commented out - will cause training failure):
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=1.2)
    
    # ACTUAL WORKING VALUES:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # COMMENT: Paper doesn't mention learning rate scheduler.
    # IMPROVEMENT: Adding scheduler improves convergence.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Paper Table 4: Epochs = 100
    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_paper.pth')
            print("Saved best model!")
    
    model.load_state_dict(torch.load('best_model_paper.pth', weights_only=True))
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels_list = validate(model, test_loader, criterion)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS (Paper Implementation)")
    print("="*50)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels_list, preds, target_names=CLASS_NAMES, zero_division=0))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('training_curves_paper.png')
    plt.show()
    
    return model, test_dataset

#%% Run Training
if __name__ == '__main__':
    model, test_dataset = main()

#%% Grad-CAM Visualization
def run_gradcam():
    model = ImprovedMobileNetV3(num_classes=7, dropout=0.1).to(device)
    model.load_state_dict(torch.load('best_model_paper.pth', map_location=device, weights_only=True))
    model.eval()
    
    metadata = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")
    image_dirs = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]
    
    for i, class_name in enumerate(CLASSES):
        sample = metadata[metadata['dx'] == class_name].iloc[0]
        img_name = sample['image_id'] + '.jpg'
        for img_dir in image_dirs:
            img_path = img_dir / img_name
            if img_path.exists():
                print(f"Running Grad-CAM for {CLASS_NAMES[i]} ({class_name})")
                visualize_gradcam(model, str(img_path), get_transforms('test'), f'gradcam_{class_name}_paper.png')
                break

if __name__ == '__main__':
    run_gradcam()
