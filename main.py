#%% Imports
import os
from dotenv import load_dotenv

# You need to make your own .env file with your Kaggle API key to run this!
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
# Of course data inspection and similar things were done, 
# but for clarity it's nice to just have one long interactive python notebook
# where this kind of exploratory code is not present. 
# The exploratory code was similar in nature as a notebook Alexander
# wrote roughly two years ago: https://github.com/AlexanderAKM/Skin_Cancer_Detection/blob/main/skin_cancer_detection.ipynb
# This python file is intended to be very clear. 

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 
               'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']

print("Downloading HAM10000 dataset via kagglehub...")
DATA_DIR = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
print(f"Dataset path: {DATA_DIR}")

#%% Transforms - as specified in paper Table 2 and Table 4
# TODO: could be possible that resizing to (192,256) contradicts the pre-training of MobileNetV3,

def get_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.8, 1.5)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.Resize((192, 256)),
            transforms.ToTensor()
        ])
    if phase == 'val':
        return transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor()
        ])
    if phase == 'test':
        return transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor()
        ])
    


#%% Dataset
class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

#%% ECA Module (Efficient Channel Attention)
class ECAModule(nn.Module):
    # Channel Attention is just like any attention method:
    # It learns which channels are important for the specific image and amplifies them.
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
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

#%% Add Dilation to Late Stages
def add_dilation(model):
    # Dilation essentially adds space inside the kernel. 
    # For instance a 3x3 with dilation=2 becomes 5x5 using the same 9 weights.
    features = model.features
    for i in range(-3, 0):
        block = features[i]
        if hasattr(block, 'block'):
            for layer in block.block:
                if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] > 1:
                    layer.dilation = (2, 2)
                    layer.padding = (2, 2)
    return model

#%% Improved MobileNet-V3
class ImprovedMobileNetV3(nn.Module):
    def __init__(self, num_classes=7, dropout=0.1):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = replace_se_with_eca(self.backbone)
        self.backbone = add_dilation(self.backbone)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self):
        return self.backbone.features

#%% Grad-CAM for Interpretability
# TODO: grad-cams all look very similar - verify correctness! perhaps use package by jacobgil
class GradCAM:
    def __init__(self, model, target_layer):
        # We register hooks to intercept values during forward and backward passes.
        # This gives information as to *where* the model "looks".
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

#%% Main training pipeline
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
    
    
    # Paper uses 70/15/15 split but doesn't mention patient-level splitting.
    # My guess is they're leaking here, but ofc hard to know, since...
    # They didn't publish their code :)
    # TODO: remove patient level splitting in our shit model and compare to our model which uses splitting
    # HAM10000 has multiple images per lesion - proper split should be by lesion_id.
    # Basically we're doing it good now (I think): This prevents data leakage where same lesion appears in train and test.
    unique_lesions = list(set(lesion_ids))
    lesion_to_indices = {}
    for idx, lid in enumerate(lesion_ids):
        if lid not in lesion_to_indices:
            lesion_to_indices[lid] = []
        lesion_to_indices[lid].append(idx)
    
    # 70% train, 15% val, 15% test
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
    
    # Weighted sampling for class balance
    # Perhaps the paper also does this for the validation and testing data sets?
    # That would also explain their super good results. But we won't know...
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create datasets
    train_dataset = SkinCancerDataset(train_paths, train_labels, get_transforms('train'))
    val_dataset = SkinCancerDataset(val_paths, val_labels, get_transforms('val'))
    test_dataset = SkinCancerDataset(test_paths, test_labels, get_transforms('test'))
    
    BATCH_SIZE = 32
    # TODO: batch size 8 is used in the paper.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ImprovedMobileNetV3(num_classes=7, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    # TODO: use either learning rate 0.2 or 0.0002 - the paper has a typo here.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
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
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")
    
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels_list = validate(model, test_loader, criterion)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
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
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, test_dataset

#%% Run Training (just comment out if you just want evaluation)
if __name__ == '__main__':
    model, test_dataset = main()

#%% EVALUATION ONLY - Load saved model and generate report results
def evaluate_saved_model():
    """Load best_model.pth and generate all results needed for a report."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    metadata_path = DATA_DIR / "HAM10000_metadata.csv"
    image_dirs = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]
    metadata = pd.read_csv(metadata_path)
    label_map = {c: i for i, c in enumerate(CLASSES)}
    
    image_paths, labels, lesion_ids = [], [], []
    for _, row in metadata.iterrows():
        img_name = row['image_id'] + '.jpg'
        for img_dir in image_dirs:
            img_path = img_dir / img_name
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(label_map[row['dx']])
                lesion_ids.append(row['lesion_id'])
                break
    
    unique_lesions = list(set(lesion_ids))
    lesion_to_indices = {lid: [] for lid in unique_lesions}
    for idx, lid in enumerate(lesion_ids):
        lesion_to_indices[lid].append(idx)
    
    train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.30, random_state=42)
    val_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.50, random_state=42)
    
    test_indices = [idx for lid in test_lesions for idx in lesion_to_indices[lid]]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    test_dataset = SkinCancerDataset(test_paths, test_labels, get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ImprovedMobileNetV3(num_classes=7, dropout=0.1).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, true_labels = validate(model, test_loader, criterion)
    
    print(f"Test Samples: {len(test_labels)}")
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")
    
    # Grad-CAM for each class
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*60)
    for i, class_name in enumerate(CLASSES):
        sample = metadata[metadata['dx'] == class_name].iloc[0]
        img_name = sample['image_id'] + '.jpg'
        for img_dir in image_dirs:
            img_path = img_dir / img_name
            if img_path.exists():
                print(f"  {CLASS_NAMES[i]}...")
                visualize_gradcam(model, str(img_path), get_transforms('test'), f'gradcam_images/gradcam_{class_name}.png')
                break
    

    print(f"""
Results on HAM10000 Test Set (n={len(test_labels)}):
- Overall Accuracy: {test_acc*100:.2f}%
- Weighted Precision: {test_prec*100:.2f}%
- Weighted Recall: {test_rec*100:.2f}%
- Weighted F1-Score: {test_f1*100:.2f}%

""")
    
    return model, test_acc, test_prec, test_rec, test_f1

#%% Run this for evaluation
if __name__ == '__main__':
    model, acc, prec, rec, f1 = evaluate_saved_model()
