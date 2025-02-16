#Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score

# Directory paths
potsdam_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Images'
potsdam_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Potsdam\Labels'
vaihingen_img_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Images'
vaihingen_mask_dir = r'C:\Users\ce841228\Documents\Dissertation\DataSet\Vaihingen\Labels'

# Segmentation classes and colors
seg_classes = [
    "Background", "Impervious surfaces", "Road", "Building", "Tree", "Low vegetation", "Car"
]
color_rgb = [
    (0, 0, 0),          # Black
    (255, 0, 0),        # Red
    (255, 255, 255),    # White
    (0, 0, 255),        # Blue
    (0, 255, 0),        # Green
    (0, 255, 255),      # Cyan
    (255, 255, 0)       # Yellow
]

pred_color_rgb = [(0, 0, 0), (60, 16, 152), (110, 193, 228), (196, 77, 255), (254, 221, 58), (21, 128, 0), (232, 98, 60)]
#==============================================================================

# Loading all image and mask file names
potsdam_files = [(f, f.replace('_RGB', '_label')) for f in os.listdir(potsdam_img_dir) if f.endswith('.tif')]
vaihingen_files = [(f, f.replace('_RGB', '_label')) for f in os.listdir(vaihingen_img_dir) if f.endswith('.tif')]

all_files = potsdam_files + vaihingen_files

# Counting the number of images and masks
num_images = len(all_files)
print(f"Total number of image-mask pairs: {num_images}")

# Separate images and masks
image_filenames = [f[0] for f in all_files]
mask_filenames = [f[1] for f in all_files]

# Checking for missing files
missing_images = [f for f in image_filenames if not os.path.exists(os.path.join(potsdam_img_dir, f)) and not os.path.exists(os.path.join(vaihingen_img_dir, f))]
missing_masks = [f for f in mask_filenames if not os.path.exists(os.path.join(potsdam_mask_dir, f)) and not os.path.exists(os.path.join(vaihingen_mask_dir, f))]
print(f"Missing Images: {len(missing_images)}")
print(f"Missing Masks: {len(missing_masks)}")

# Removing pairs with missing data
valid_files = [(img, mask) for img, mask in all_files if img not in missing_images and mask not in missing_masks]
print(f"Total valid image-mask pairs: {len(valid_files)}")
#==============================================================================

# Load images and masks
def load_img(path, channels=3):
    img = Image.open(path)
    return np.array(img.convert('RGB') if channels == 3 else img.convert('L'))

def norm_colors(img_array):
    return img_array.astype(np.float32) / 255.0

def resize_image(img_array, mask_array, target_size):
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_array, target_size, interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized

def class_to_rgb(mask, color_map):
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        rgb_image[mask == i] = np.array(color)
    return rgb_image

# One-hot encode mask array
def one_hot_encode_mask(mask, num_classes):
    height, width = mask.shape[:2]
    class_indices = np.zeros((height, width), dtype=np.int32)
    
    # Ensuring mask is in RGB format for correct comparison
    mask = np.array(mask)
    
    # Applied one-hot encoding by comparing each pixel with the class colors
    for i, color in enumerate(color_rgb):
        class_indices[np.all(mask == color, axis=-1)] = i
    
    # Return one-hot encoded version of the class indices
    return tf.one_hot(class_indices, num_classes)

def get_data(img_file, mask_file, img_dir, mask_dir, target_size=None, num_classes=7):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    # Load image and mask
    img_array = load_img(img_path, channels=3)
    mask_array = load_img(mask_path, channels=3)
    
    # Normalize image
    img_array = norm_colors(img_array)
    
    # Resize if target size is provided
    if target_size:
        img_array, mask_array = resize_image(img_array, mask_array, target_size)
    
    # One-hot encode the mask
    mask_array = one_hot_encode_mask(mask_array, num_classes)
    
    return img_array, mask_array

# Used the `valid_files` after filtering missing/corrupt files
data = []
for img_file, mask_file in valid_files:
    img_dir = potsdam_img_dir if img_file in os.listdir(potsdam_img_dir) else vaihingen_img_dir
    mask_dir = potsdam_mask_dir if mask_file in os.listdir(potsdam_mask_dir) else vaihingen_mask_dir
    img_array, mask_array = get_data(img_file, mask_file, img_dir, mask_dir, target_size=(256, 256))
    data.append({'image_filename': img_file, 'mask_filename': mask_file, 'image_array': img_array, 'mask_array': mask_array})

df = pd.DataFrame(data)
#==============================================================================
# EDA
# 1. Visualize random samples from the dataset
def visualize_random_samples(data, num_samples=5):
    for i in np.random.choice(len(data), num_samples, replace=False):
        img_array = data.iloc[i]['image_array']
        mask_array = np.argmax(data.iloc[i]['mask_array'], axis=-1)
        
        # Convert mask to RGB for better visualization
        mask_rgb = class_to_rgb(mask_array, color_rgb)

        # Plot the original image and mask
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img_array)
        ax1.set_title(f"Image: {data.iloc[i]['image_filename']}")
        
        ax2.imshow(mask_rgb)
        ax2.set_title(f"Mask: {data.iloc[i]['mask_filename']}")
        
        plt.tight_layout()
        plt.show()

# Visualize random samples
visualize_random_samples(df, num_samples=5)

# 2. Calculate and Plot Class Distribution
def calculate_class_distribution(masks, num_classes):
    class_distribution = np.zeros(num_classes, dtype=np.int32)
    for mask in masks:
        # Convert one-hot encoded mask to class index
        mask_class_indices = np.argmax(mask, axis=-1)
        # Count the number of pixels for each class
        unique, counts = np.unique(mask_class_indices, return_counts=True)
        class_distribution[unique] += counts
    return class_distribution

def plot_class_distribution(class_distribution, seg_classes, title='Class Distribution'):
    plt.figure(figsize=(10, 6))
    
    # Creating a bar chart
    bars = plt.bar(range(len(seg_classes)), class_distribution, color='skyblue')
    
    # Adding text labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    # Set labels, title, and ticks
    plt.xticks(range(len(seg_classes)), seg_classes, rotation=45, ha="right")
    plt.ylabel('Pixel Count')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Calculate class distribution for the entire dataset
all_masks = np.stack(df['mask_array'].values)
class_distribution = calculate_class_distribution(all_masks, num_classes=len(seg_classes))

# Plot class distribution
plot_class_distribution(class_distribution, seg_classes, title='Pixel based Class Distribution in the Entire Dataset')

# 3. Checking Image and Mask Shape Distribution
def analyze_image_shapes(data):
    image_shapes = [img.shape for img in data['image_array']]
    mask_shapes = [mask.shape for mask in data['mask_array']]
    
    # Convert to DataFrame for analysis
    img_shapes_df = pd.DataFrame(image_shapes, columns=['Height', 'Width', 'Channels'])
    mask_shapes_df = pd.DataFrame(mask_shapes, columns=['Height', 'Width', 'Classes'])

    print(f"Unique image shapes: {img_shapes_df['Height'].unique()} x {img_shapes_df['Width'].unique()}")
    print(f"Unique mask shapes: {mask_shapes_df['Height'].unique()} x {mask_shapes_df['Width'].unique()}")

# Analyze image and mask shape distributions
analyze_image_shapes(df)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 for validation
#==============================================================================
# U-Net model
#==============================================================================
def unet_model(input_size=(256, 256, 3), num_classes=7):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder (Downsampling)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder (Upsampling)
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
#==============================================================================
# Categorical Cross-Entropy Loss
def cce_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# IoU metric
def iou_metric(y_true, y_pred):
    # Convert one-hot encoded masks back to class indices
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.greater(y_true, 0), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.greater(y_true, 0) | tf.greater(y_pred, 0), tf.float32))

    iou = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
    return iou

# Precision metric
def precision_metric(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.greater(y_true, 0), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.greater(y_pred, 0), tf.float32))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

# Recall metric
def recall_metric(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.greater(y_true, 0), tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(tf.greater(y_true, 0), tf.float32))
    
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall

# F1-score metric
def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_score

# Callback to format output during training
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')
        train_iou = logs.get('iou_metric')
        train_precision = logs.get('precision_metric')
        train_recall = logs.get('recall_metric')
        train_f1 = logs.get('f1_metric')
        
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        val_iou = logs.get('val_iou_metric')
        val_precision = logs.get('val_precision_metric')
        val_recall = logs.get('val_recall_metric')
        val_f1 = logs.get('val_f1_metric')

        print(f"Epoch {epoch+1}:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy*100:.2f}%, IoU: {train_iou*100:.2f}%, "
              f"Precision: {train_precision*100:.2f}%, Recall: {train_recall*100:.2f}%, F1: {train_f1*100:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy*100:.2f}%, IoU: {val_iou*100:.2f}%, "
              f"Precision: {val_precision*100:.2f}%, Recall: {val_recall*100:.2f}%, F1: {val_f1*100:.2f}%\n")

#==============================================================================
# Compile and train the model
def train_model(train_df, val_df, num_classes=7, batch_size=32, epochs=75):
    model = unet_model(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Added custom metrics to the compilation
    model.compile(optimizer=optimizer, loss=cce_loss, metrics=[
        'accuracy', 
        iou_metric, 
        precision_metric, 
        recall_metric, 
        f1_metric
    ])

    # Training the model
    history = model.fit(
        x=np.stack(train_df['image_array'].values),
        y=np.stack(train_df['mask_array'].values),
        validation_data=(np.stack(val_df['image_array'].values), np.stack(val_df['mask_array'].values)),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[CustomCallback(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    
    return model, history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Model evaluation on test set
def evaluate_model(model, test_df, num_classes=7):
    test_images = np.stack(test_df['image_array'].values)
    test_masks = np.stack(test_df['mask_array'].values)
    
    # Evaluate model performance
    metrics = model.evaluate(test_images, test_masks, verbose=0)
    
    # Unpack the metrics
    loss = metrics[0]
    accuracy = metrics[1]
    iou = metrics[2]
    precision = metrics[3]
    recall = metrics[4]
    f1 = metrics[5]
    
    # Print the results
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test IoU: {iou*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    
    # Predictions
    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    
    return y_pred, y_pred_classes
#==============================================================================
# Visualize predictions for a few test images
def visualize_predictions(model, test_df, seg_classes, color_rgb, pred_color_rgb, num_samples=5):
    for i in range(num_samples):
        # Load image and true mask
        image = test_df['image_array'].iloc[i]
        true_mask = test_df['mask_array'].iloc[i]

        # Predict mask from the model
        prediction = model.predict(np.expand_dims(image, axis=0))[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        # Convert true mask and predicted mask to RGB format using the provided color maps
        true_mask_rgb = class_to_rgb(np.argmax(true_mask, axis=-1), color_rgb)
        predicted_mask_rgb = class_to_rgb(predicted_mask, pred_color_rgb)

        # Plot the original image, true mask, and predicted mask
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.imshow(image)
        ax1.set_title('Original Image')

        ax2.imshow(true_mask_rgb)
        ax2.set_title('True Mask')

        ax3.imshow(predicted_mask_rgb)
        ax3.set_title('Predicted Mask')

        # Add legend for true mask segmentation classes
        true_legend_handles = [mpatches.Patch(color=np.array(color_rgb[idx])/255, label=class_name)
                               for idx, class_name in enumerate(seg_classes)]
        ax2.legend(handles=true_legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Add legend for predicted mask segmentation classes
        pred_legend_handles = [mpatches.Patch(color=np.array(pred_color_rgb[idx])/255, label=class_name)
                               for idx, class_name in enumerate(seg_classes)]
        ax3.legend(handles=pred_legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.show()

# Train the model
model, history = train_model(train_df, val_df)

# Plot training history
plot_training_history(history)

# Evaluate model on test set
y_pred, y_pred_classes = evaluate_model(model, test_df, num_classes=len(seg_classes))

# Visualize predictions for the first 5 test images
visualize_predictions(model, test_df, seg_classes, color_rgb, pred_color_rgb, num_samples=5)