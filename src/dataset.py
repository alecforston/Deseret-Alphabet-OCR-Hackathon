import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path


class DeseretDataset(Dataset):
    """Dataset for training/validation"""
    
    def __init__(self, image_dir, label_dir, char_to_idx, 
                 target_height=64, max_width=1024, augment=False):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing label text files
            char_to_idx: Dictionary mapping characters to indices
            target_height: Target height for images
            max_width: Maximum width for images
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.char_to_idx = char_to_idx
        self.target_height = target_height
        self.max_width = max_width
        self.augment = augment
        
        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.glob('*.png')))
        
        # Filter to only images with corresponding labels
        self.valid_paths = []
        for img_path in self.image_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.valid_paths.append(img_path)
        
        print(f"Loaded {len(self.valid_paths)} image-label pairs")
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        img_path = self.valid_paths[idx]
        
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Preprocess image
        img = self.preprocess_image(img)
        
        # Load label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        with open(label_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Convert text to indices
        label = []
        for char in text:
            if char in self.char_to_idx:
                label.append(self.char_to_idx[char])
            else:
                # Unknown character - skip or use special token
                print(f"Warning: Unknown character '{char}' in {img_path.stem}")
        
        # Convert to tensors
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        label_tensor = torch.LongTensor(label)
        
        return img_tensor, label_tensor, len(label)
    
    def preprocess_image(self, img):
        """Preprocess a single image"""
        # Trim whitespace
        thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            img = img[y:y+h, x:x+w]
        
        # Resize maintaining aspect ratio
        h, w = img.shape
        scale = self.target_height / h
        new_w = int(w * scale)
        new_h = self.target_height
        
        if new_w > self.max_width:
            scale = self.max_width / w
            new_w = self.max_width
            new_h = int(h * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self.augment_image(img)
        
        # Pad to target dimensions
        final_img = np.ones((self.target_height, self.max_width), dtype=np.float32)
        y_offset = (self.target_height - new_h) // 2
        final_img[y_offset:y_offset+new_h, :new_w] = img
        
        # Normalize to [0, 1]
        final_img = final_img / 255.0
        
        return final_img
    
    def augment_image(self, img):
        """Apply random augmentation"""
        # Random brightness adjustment
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Random slight rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-2, 2)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        
        # Random Gaussian noise
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 3, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img


class DeseretTestDataset(Dataset):
    """Dataset for test/inference"""
    
    def __init__(self, image_dir, target_height=64, max_width=1024):
        """
        Args:
            image_dir: Directory containing test images
            target_height: Target height for images
            max_width: Maximum width for images
        """
        self.image_dir = Path(image_dir)
        self.target_height = target_height
        self.max_width = max_width
        
        self.image_paths = sorted(list(self.image_dir.glob('*.png')))
        print(f"Loaded {len(self.image_paths)} test images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Preprocess image
        img = self.preprocess_image(img)
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
        # Return image and ID
        return img_tensor, img_path.stem
    
    def preprocess_image(self, img):
        """Preprocess a single image (same as training)"""
        # Trim whitespace
        thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            img = img[y:y+h, x:x+w]
        
        # Resize maintaining aspect ratio
        h, w = img.shape
        scale = self.target_height / h
        new_w = int(w * scale)
        new_h = self.target_height
        
        if new_w > self.max_width:
            scale = self.max_width / w
            new_w = self.max_width
            new_h = int(h * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target dimensions
        final_img = np.ones((self.target_height, self.max_width), dtype=np.float32)
        y_offset = (self.target_height - new_h) // 2
        final_img[y_offset:y_offset+new_h, :new_w] = img
        
        # Normalize to [0, 1]
        final_img = final_img / 255.0
        
        return final_img


def collate_fn(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: List of (image, label, label_length) tuples
    
    Returns:
        images: (batch, 1, H, W)
        labels: Concatenated labels
        label_lengths: Length of each label
    """
    images, labels, label_lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Concatenate labels
    labels = torch.cat(labels, 0)
    
    # Label lengths
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, labels, label_lengths