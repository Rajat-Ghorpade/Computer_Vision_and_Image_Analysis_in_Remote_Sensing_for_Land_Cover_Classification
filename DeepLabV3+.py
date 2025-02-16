import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from segmentation_models_pytorch import DeepLabV3Plus, losses
from bayes_opt import BayesianOptimization
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

print("All packages are successfully imported!")

# Set the GPU to use
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("GPU is being used")
    except RuntimeError as e:
        print(e)

# Define directory paths
potsdam_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Images'
potsdam_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Labels'
vaihingen_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Images'
vaihingen_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Labels'

# Define segmentation classes and colors
seg_classes = ["Background", "Impervious surfaces", "Road", "Building", "Tree", "Low vegetation", "Car"]
true_color_rgb = [(0, 0, 0), (255, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]

pred_color_rgb = [(0, 0, 0), (60, 16, 152), (110, 193, 228), (196, 77, 255), (254, 221, 58), (21, 128, 0), (232, 98, 60)]

ISPRS_colors = pd.DataFrame(data={"Classes": seg_classes, "Color_RGB": true_color_rgb})
print(ISPRS_colors)

def rgb_to_class(mask, color_map):
    mask_class = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i, color in enumerate(color_map):
        mask_class[np.all(mask == color, axis=-1)] = i
    return mask_class

def class_to_rgb(mask_class, color_map):
    mask_rgb = np.zeros((*mask_class.shape, 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        mask_rgb[mask_class == i] = color
    return mask_rgb

class PotsdamVaihingenDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, labeled=True, color_map=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.labeled = labeled
        self.color_map = color_map
        self.target_size = target_size
        self.images = sorted(os.listdir(image_dir))
        if self.labeled:
            self.masks = sorted(os.listdir(mask_dir))
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.labeled:
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = Image.open(mask_path)
            mask = mask.resize(self.target_size, Image.NEAREST)
            mask = np.array(mask)
            mask = rgb_to_class(mask, self.color_map)
            mask = torch.tensor(mask, dtype=torch.long)
            if self.transform:
                image = self.transform(image)
            return image, mask
        else:
            if self.transform:
                image = self.transform(image)
            return image

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

# Define the DeepLabV3+ model with ResNet-101 backbone
class DeepLabV3PlusModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3PlusModel, self).__init__()
        self.model = DeepLabV3Plus(
            encoder_name="resnet101",  
            encoder_weights="imagenet",  
            in_channels=3,  
            classes=num_classes, 
            activation=None  
        )

    def forward(self, x):
        return self.model(x)

def calculate_metrics(pred, target, num_classes):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    precision = precision_score(target_flat, pred_flat, average='macro', zero_division=0) * 100
    recall = recall_score(target_flat, pred_flat, average='macro', zero_division=0) * 100
    f1 = f1_score(target_flat, pred_flat, average='macro', zero_division=0) * 100
    iou = jaccard_score(target_flat, pred_flat, average='macro') * 100

    return precision, recall, f1, iou

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_precision, total_recall, total_f1, total_iou = 0, 0, 0, 0

    for img_labeled, mask_labeled in dataloader:
        img_labeled, mask_labeled = img_labeled.to(device), mask_labeled.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(img_labeled)
        loss = criterion(pred, mask_labeled)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred_classes = torch.argmax(pred, dim=1)
        correct += (pred_classes == mask_labeled).sum().item()
        total += mask_labeled.numel()

        precision, recall, f1, iou = calculate_metrics(pred_classes.cpu().numpy(), mask_labeled.cpu().numpy(), num_classes)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_iou += iou

    accuracy = correct / total * 100
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return total_loss / len(dataloader), accuracy, avg_precision, avg_recall, avg_f1, avg_iou

def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_precision, total_recall, total_f1, total_iou = 0, 0, 0, 0

    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            total_loss += loss.item()

            pred_classes = torch.argmax(output, dim=1)
            correct += (pred_classes == mask).sum().item()
            total += mask.numel()

            precision, recall, f1, iou = calculate_metrics(pred_classes.cpu().numpy(), mask.cpu().numpy(), num_classes)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_iou += iou

    accuracy = correct / total * 100
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return total_loss / len(dataloader), accuracy, avg_precision, avg_recall, avg_f1, avg_iou

def test(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_precision, total_recall, total_f1, total_iou = 0, 0, 0, 0

    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            total_loss += loss.item()

            pred_classes = torch.argmax(output, dim=1)
            correct += (pred_classes == mask).sum().item()
            total += mask.numel()

            precision, recall, f1, iou = calculate_metrics(pred_classes.cpu().numpy(), mask.cpu().numpy(), num_classes)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_iou += iou

    accuracy = correct / total * 100
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return total_loss / len(dataloader), accuracy, avg_precision, avg_recall, avg_f1, avg_iou

def visualize_predictions(model, dataloader, true_color_map, pred_color_map, device):
    model.eval()
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(device)
            pred = model(img)
            pred_class = torch.argmax(pred, dim=1).cpu().numpy()
            img = img.cpu().numpy().transpose(0, 2, 3, 1)
            mask = mask.cpu().numpy()
            pred_rgb = [class_to_rgb(pc, pred_color_map) for pc in pred_class]
            true_rgb = [class_to_rgb(mc, true_color_map) for mc in mask]

            for i in range(len(img)):
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(img[i])
                plt.title("Original Image")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(true_rgb[i])
                plt.title("True Mask")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(pred_rgb[i])
                plt.title("Predicted Mask")
                plt.axis("off")

                handles_true = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255.0) for c in true_color_map]
                handles_pred = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255.0) for c in pred_color_map]
                labels = seg_classes
                plt.legend(handles_true, labels, title="True Mask", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend(handles_pred, labels, title="Predicted Mask", bbox_to_anchor=(1.05, 0.5), loc='upper left')

                plt.show()

def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, marker='o', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Combine Potsdam and Vaihingen datasets
    potsdam_dataset = PotsdamVaihingenDataset(potsdam_img_dir, potsdam_mask_dir, transform=get_transform(), color_map=true_color_rgb)
    vaihingen_dataset = PotsdamVaihingenDataset(vaihingen_img_dir, vaihingen_mask_dir, transform=get_transform(), color_map=true_color_rgb)
    full_dataset = ConcatDataset([potsdam_dataset, vaihingen_dataset])

    # Split the combined dataset into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    num_classes = len(seg_classes)
    model = DeepLabV3PlusModel(num_classes=num_classes).to(device)

    # Criterion (Use Dice Loss or IoU Loss)
    criterion = losses.DiceLoss(mode='multiclass')

    # Optimizer with Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    # Hyperparameter Tuning using Bayesian Optimization
    def objective_function(lr, weight_decay):
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = weight_decay
        
        # Perform one epoch of training and validation
        train_loss, _, _, _, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, _, _, _, _, _ = validate(model, val_loader, criterion, device, num_classes)
        
        return -val_loss  # Maximizing negative validation loss means minimizing validation loss

    pbounds = {'lr': (1e-5, 1e-3),'weight_decay': (1e-5, 1e-3)}

    optimizer_bayes = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42, verbose=2)

    optimizer_bayes.maximize(init_points=5, n_iter=20)

    # Apply the best hyperparameters
    best_hps = optimizer_bayes.max['params']
    optimizer.param_groups[0]['lr'] = best_hps['lr']
    optimizer.param_groups[0]['weight_decay'] = best_hps['weight_decay']
    print(f"Best Hyperparameters: {best_hps}")

    # Training loop
    num_epochs = 75
    train_losses, train_accuracies, train_precisions, train_recalls, train_f1s, train_ious = [], [], [], [], [], []
    val_losses, val_accuracies, val_precisions, val_recalls, val_f1s, val_ious = [], [], [], [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_prec, train_recall, train_f1, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_acc, val_prec, val_recall, val_f1, val_iou = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_precisions.append(train_prec)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_ious.append(train_iou)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_ious.append(val_iou)

        print(f'Epoch {epoch + 1}/{num_epochs}, Time: {epoch_time:.2f}s, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, '
              f'Precision: {val_prec:.2f}%, Recall: {val_recall:.2f}%, F1-Score: {val_f1:.2f}%, IoU: {val_iou:.2f}%')

    # Test the model
    test_loss, test_acc, test_prec, test_recall, test_f1, test_iou = test(model, test_loader, criterion, device, num_classes)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, '
          f'Precision: {test_prec:.2f}%, Recall: {test_recall:.2f}%, F1-Score: {test_f1:.2f}%, IoU: {test_iou:.2f}%')

    # Visualize predictions on validation set
    visualize_predictions(model, val_loader, true_color_rgb, pred_color_rgb, device)

    # Plot the results
    plot_results(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == '__main__':
    main()