import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
DEFECT_CLASSES = ['no_defect', 'scratch', 'dent', 'color_variation', 'misalignment']
NUM_CLASSES = len(DEFECT_CLASSES)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DATA_DIR = "dataset/defects"  # Update with your dataset path
MODEL_SAVE_PATH = "output/defect_model.pth"

class DefectDataset(Dataset):
    """Dataset for defect detection training"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data():
    """Load image paths and labels from dataset directory.
    
    Expected structure:
    DATA_DIR/
        no_defect/
            img1.jpg
            img2.jpg
            ...
        scratch/
            img1.jpg
            ...
        ...
    """
    image_paths = []
    labels = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(DEFECT_CLASSES)}
    
    # Check if directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Warning: {DATA_DIR} does not exist. Please create it and add your defect images.")
        print("You'll need images organized in subfolders according to defect type.")
        print("Creating empty directory for now...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for cls_name in DEFECT_CLASSES:
            os.makedirs(os.path.join(DATA_DIR, cls_name), exist_ok=True)
        
        # Return empty lists
        return [], []
    
    # Load images from each class directory
    for cls_name in DEFECT_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir, exist_ok=True)
            continue
            
        cls_idx = class_to_idx[cls_name]
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                image_paths.append(img_path)
                labels.append(cls_idx)
    
    return image_paths, labels

def get_data_transforms():
    """Define image transformations for training and validation"""
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transformations
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model():
    """Create and initialize the defect detection model"""
    # Load pre-trained EfficientNet model
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    return model

def train_model():
    """Train the defect detection model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    image_paths, labels = prepare_data()
    
    if len(image_paths) == 0:
        print("No images found. Please add images to the dataset directory.")
        return
    
    print(f"Found {len(image_paths)} images for training.")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = DefectDataset(X_train, y_train, transform=train_transform)
    val_dataset = DefectDataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = create_model()
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """Plot training history"""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_history.png')
    plt.close()

def generate_sample_data(num_samples=100):
    """Generate sample data for training if no real data is available.
    
    This is just for demonstration purposes, in a real application
    you should use real defect images.
    """
    print("Generating sample data for demonstration...")
    
    # Create directories
    for cls_name in DEFECT_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
    
    # For each class, generate sample images
    for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        num_cls_samples = num_samples // NUM_CLASSES
        
        for i in range(num_cls_samples):
            # Create a base image
            img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 200
            
            # Add class-specific patterns
            if cls_name == 'scratch':
                # Add random scratches
                for _ in range(random.randint(1, 3)):
                    x1 = random.randint(0, IMAGE_SIZE-1)
                    y1 = random.randint(0, IMAGE_SIZE-1)
                    x2 = random.randint(0, IMAGE_SIZE-1)
                    y2 = random.randint(0, IMAGE_SIZE-1)
                    cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
            
            elif cls_name == 'dent':
                # Add random dents
                for _ in range(random.randint(1, 2)):
                    x = random.randint(20, IMAGE_SIZE-20)
                    y = random.randint(20, IMAGE_SIZE-20)
                    radius = random.randint(10, 30)
                    cv2.circle(img, (x, y), radius, (150, 150, 150), -1)
            
            elif cls_name == 'color_variation':
                # Add color patches
                for _ in range(random.randint(1, 3)):
                    x = random.randint(0, IMAGE_SIZE-50)
                    y = random.randint(0, IMAGE_SIZE-50)
                    w = random.randint(30, 60)
                    h = random.randint(30, 60)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            
            elif cls_name == 'misalignment':
                # Add misaligned elements
                for _ in range(random.randint(1, 2)):
                    x = random.randint(0, IMAGE_SIZE-100)
                    y = random.randint(0, IMAGE_SIZE-100)
                    cv2.rectangle(img, (x, y), (x+80, y+80), (200, 200, 200), -1)
                    shift = random.randint(10, 30)
                    cv2.rectangle(img, (x+shift, y+shift), (x+shift+60, y+shift+60), (100, 100, 100), -1)
            
            # Save image
            img_path = os.path.join(cls_dir, f"sample_{i}.jpg")
            cv2.imwrite(img_path, img)
    
    print(f"Generated {num_samples} sample images in {DATA_DIR}")

if __name__ == "__main__":
    # Check if dataset exists, if not, generate sample data
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print("Dataset not found. Generating sample data for demonstration.")
        generate_sample_data()
    
    # Train the model
    model = train_model() 