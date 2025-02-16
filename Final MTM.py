import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import copy
from tqdm import tqdm
from torch.nn import functional as F
from segmentation_models_pytorch import DeepLabV3Plus
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
import matplotlib.patches as mpatches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths, classes, colors
potsdam_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Images'
potsdam_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Labels'
vaihingen_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Images'
vaihingen_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Labels'

seg_classes = ["Background", "Impervious surfaces", "Road", "Building", "Tree", "Low vegetation", "Car"]
color_rgb = [(0, 0, 0), (255, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0)]
pred_color_rgb = [(0, 0, 0), (60, 16, 152), (110, 193, 228), (196, 77, 255), (254, 221, 58), (21, 128, 0), (232, 98, 60)]
num_classes = len(seg_classes)

# Utility functions for data handling
def rgb_to_class(mask_rgb, color_mapping):
    class_map = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    for idx, color in enumerate(color_mapping):
        class_map[np.all(mask_rgb == color, axis=-1)] = idx
    return class_map

def load_img(path, channels=3):
    img = Image.open(path)
    return np.array(img.convert('RGB') if channels == 3 else img.convert('L'))

def resize_and_normalize(img_array, mask_array, target_size=(256, 256)):
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_array, target_size, interpolation=cv2.INTER_NEAREST)
    return img_resized.astype(np.float32) / 255.0, mask_resized

class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = torch.tensor(data['image_array'], dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(data['mask_array'], dtype=torch.long)
        return image, mask

    def __len__(self):
        return len(self.df)

# Mean Teacher Model
class MeanTeacherModel(nn.Module):
    def __init__(self, ema_decay=0.995):
        super(MeanTeacherModel, self).__init__()
        self.student = DeepLabV3Plus(encoder_name='resnet50', classes=num_classes)
        self.teacher = copy.deepcopy(self.student)
        self.ema_decay = ema_decay
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_output = self.student(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        return student_output, teacher_output

    def update_teacher_weights(self):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)

# Data loading and preprocessing
def load_data(img_dir, mask_dir):
    files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    data = []
    for f in files:
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f.replace('_RGB', '_label'))
        img_array, mask_array = load_img(img_path), load_img(mask_path)
        img_array, mask_array = resize_and_normalize(img_array, mask_array)
        mask_array = rgb_to_class(mask_array, color_rgb)
        data.append({'image_array': img_array, 'mask_array': mask_array})
    return pd.DataFrame(data)

# Combine datasets and split
potsdam_data = load_data(potsdam_img_dir, potsdam_mask_dir)
vaihingen_data = load_data(vaihingen_img_dir, vaihingen_mask_dir)
full_data = pd.concat([potsdam_data, vaihingen_data])
train_df, test_df = train_test_split(full_data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_dataset = SegmentationDataset(train_df)
val_dataset = SegmentationDataset(val_df)
test_dataset = SegmentationDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def calculate_metrics(all_targets, all_preds):
    accuracy = accuracy_score(all_targets, all_preds) * 100
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    iou = jaccard_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    return accuracy, precision, recall, f1, iou

def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs):
    metrics = {
        'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [], 'train_iou': [],
        'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': [], 'val_iou': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss, all_train_preds, all_train_targets = 0, [], []

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = F.cross_entropy(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_train_preds.extend(predicted.cpu().numpy().flatten())
            all_train_targets.extend(masks.cpu().numpy().flatten())

        train_loss = train_loss / len(train_loader)
        train_acc, train_prec, train_rec, train_f1, train_iou = calculate_metrics(all_train_targets, all_train_preds)

        model.eval()
        val_loss, all_val_preds, all_val_targets = 0, [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs, _ = model(images)
                loss = F.cross_entropy(outputs, masks)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_val_preds.extend(predicted.cpu().numpy().flatten())
                all_val_targets.extend(masks.cpu().numpy().flatten())

        val_loss = val_loss / len(val_loader)
        val_acc, val_prec, val_rec, val_f1, val_iou = calculate_metrics(all_val_targets, all_val_preds)

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_prec'].append(train_prec)
        metrics['train_rec'].append(train_rec)
        metrics['train_f1'].append(train_f1)
        metrics['train_iou'].append(train_iou)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_prec'].append(val_prec)
        metrics['val_rec'].append(val_rec)
        metrics['val_f1'].append(val_f1)
        metrics['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Prec: {train_prec:.2f}%, Rec: {train_rec:.2f}%, F1: {train_f1:.2f}%, IoU: {train_iou:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Prec: {val_prec:.2f}%, Rec: {val_rec:.2f}%, F1: {val_f1:.2f}%, IoU: {val_iou:.2f}%")

    # Print final epoch results
    print("\nFinal Epoch Results:")
    print(f"Train - Loss: {metrics['train_loss'][-1]:.4f}, Acc: {metrics['train_acc'][-1]:.2f}%, Prec: {metrics['train_prec'][-1]:.2f}%, Rec: {metrics['train_rec'][-1]:.2f}%, F1: {metrics['train_f1'][-1]:.2f}%, IoU: {metrics['train_iou'][-1]:.2f}%")
    print(f"Val   - Loss: {metrics['val_loss'][-1]:.4f}, Acc: {metrics['val_acc'][-1]:.2f}%, Prec: {metrics['val_prec'][-1]:.2f}%, Rec: {metrics['val_rec'][-1]:.2f}%, F1: {metrics['val_f1'][-1]:.2f}%, IoU: {metrics['val_iou'][-1]:.2f}%")

    return metrics

# Bayesian Optimization
def bayesian_optimization():
    def black_box_function(lr, weight_decay):
        model = MeanTeacherModel().to(device)
        optimizer = optim.Adam(model.student.parameters(), lr=lr, weight_decay=weight_decay)
        metrics = train_and_evaluate(model, train_loader, val_loader, optimizer, 10)
        return max(metrics['val_acc'])

    pbounds = {'lr': (1e-5, 1e-3), 'weight_decay': (1e-5, 1e-3)}
    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=2, n_iter=3)
    return optimizer.max['params']

best_params = bayesian_optimization()
print("Best hyperparameters found:", best_params)

# Training with the best parameters
model = MeanTeacherModel(ema_decay=0.995).to(device)
optimizer = optim.Adam(model.student.parameters(), lr=1e-4, weight_decay=best_params['weight_decay'])
metrics = train_and_evaluate(model, train_loader, val_loader, optimizer, 75)

# Plotting
plt.figure(figsize=(15, 10))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(np.arange(1, len(metrics['train_loss']) + 1) / len(metrics['train_loss']), metrics['train_loss'], label='Training Loss')
plt.plot(np.arange(1, len(metrics['val_loss']) + 1) / len(metrics['val_loss']), metrics['val_loss'], label='Validation Loss')
plt.title('Loss over Training Progress')
plt.xlabel('Training Progress (ratio)')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(np.arange(1, len(metrics['train_acc']) + 1) / len(metrics['train_acc']), metrics['train_acc'], label='Training Accuracy')
plt.plot(np.arange(1, len(metrics['val_acc']) + 1) / len(metrics['val_acc']), metrics['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Training Progress')
plt.xlabel('Training Progress (ratio)')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Test the model and print metrics
def test_model(model, test_loader):
    model.eval()
    test_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs, _ = model(images)
            loss = F.cross_entropy(outputs, masks)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    test_loss = test_loss / len(test_loader)
    test_acc, test_prec, test_rec, test_f1, test_iou = calculate_metrics(all_targets, all_preds)

    print(f'Test Results:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.2f}%')
    print(f'Precision: {test_prec:.2f}%')
    print(f'Recall: {test_rec:.2f}%')
    print(f'F1 Score: {test_f1:.2f}%')
    print(f'IoU: {test_iou:.2f}%')

    return test_loss, test_acc, test_prec, test_rec, test_f1, test_iou

test_metrics = test_model(model, test_loader)

# Visualization function
def class_to_rgb(mask, color_map):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        rgb[mask == i] = color
    return rgb

def visualize_predictions(model, loader, num_images=5):
    model.eval()
    images_processed = 0
    with torch.no_grad():
        for images, masks in loader:
            if images_processed >= num_images:
                break
            images, masks = images.to(device), masks.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_processed >= num_images:
                    break

                plt.figure(figsize=(18, 6))
                
                # Original Image
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(images[i].cpu().permute(1, 2, 0))
                ax1.set_title('Original Image')
                ax1.axis('off')

                # True Mask
                ax2 = plt.subplot(1, 3, 2)
                true_color_mask = class_to_rgb(masks[i].cpu().numpy(), color_rgb)
                ax2.imshow(true_color_mask)
                ax2.set_title('True Mask')
                ax2.axis('off')

                # Predicted Mask
                ax3 = plt.subplot(1, 3, 3)
                pred_color_mask = class_to_rgb(predicted[i].cpu().numpy(), pred_color_rgb)
                ax3.imshow(pred_color_mask)
                ax3.set_title('Predicted Mask')
                ax3.axis('off')

                # Add legends to the True Mask and Predicted Mask plots
                true_legend_handles = [mpatches.Patch(color=np.array(color_rgb[idx])/255, label=seg_classes[idx]) for idx in range(num_classes)]
                pred_legend_handles = [mpatches.Patch(color=np.array(pred_color_rgb[idx])/255, label=seg_classes[idx]) for idx in range(num_classes)]

                ax2.legend(handles=true_legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                ax3.legend(handles=pred_legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

                plt.tight_layout()
                plt.show()

                images_processed += 1

# Call the function to visualize predictions
visualize_predictions(model, test_loader, num_images=5)