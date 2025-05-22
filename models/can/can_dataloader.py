import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass

# Configuration
@dataclass
class DataConfig:
    INPUT_HEIGHT: int = 128
    INPUT_WIDTH: int = 128 * 8
    MAX_LENGTH: int = 150
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    BASE_DIR: str = 'data/CROHME'

CONFIG = DataConfig()

def is_effectively_binary(img: np.ndarray, threshold: float = 0.9) -> bool:
    """Check if image is effectively binary based on pixel intensity."""
    dark_pixels = np.sum(img < 20)
    bright_pixels = np.sum(img > 235)
    total_pixels = img.size
    return (dark_pixels + bright_pixels) / total_pixels > threshold

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Preprocess image with edge detection, cropping, and thresholding."""
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((7, 13), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=8)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # Find optimal crop
    best_f1 = 0
    best_crop = (0, 0, image.shape[1], image.shape[0])
    total_white_pixels = np.sum(dilated > 0)
    
    current_mask = np.zeros_like(dilated)
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0

    for idx in sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True):
        component_mask = (labels == idx)
        current_mask |= component_mask
        
        comp_y, comp_x = np.where(component_mask)
        if len(comp_x) > 0 and len(comp_y) > 0:
            x_min = min(x_min, np.min(comp_x))
            y_min = min(y_min, np.min(comp_y))
            x_max = max(x_max, np.max(comp_x))
            y_max = max(y_max, np.max(comp_y))
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        crop_area = width * height
        
        crop_mask = np.zeros_like(dilated)
        crop_mask[y_min:y_max+1, x_min:x_max+1] = 1
        white_in_crop = np.sum((dilated > 0) & (crop_mask > 0))
        
        precision = white_in_crop / crop_area if crop_area > 0 else 0
        recall = white_in_crop / total_white_pixels if total_white_pixels > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_crop = (x_min, y_min, x_max, y_max)
    
    # Crop image
    x_min, y_min, x_max, y_max = best_crop
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    # Thresholding
    thresh = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)[1] if is_effectively_binary(cropped) else \
             cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Ensure black background
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = 255 - thresh
    
    # Denoise
    denoised = cv2.medianBlur(thresh, 3)
    for _ in range(3):
        denoised = cv2.medianBlur(denoised, 3)
    
    # Add padding
    result = cv2.copyMakeBorder(denoised, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    return result, best_crop

def process_image(filename: str, convert_to_rgb: bool = False) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Load and process image with resizing and padding."""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {filename}")

    processed, crop = preprocess_image(image)
    
    # Resize
    h, w = processed.shape
    new_w = int((CONFIG.INPUT_HEIGHT / h) * w)
    
    if new_w > CONFIG.INPUT_WIDTH:
        resized = cv2.resize(processed, (CONFIG.INPUT_WIDTH, CONFIG.INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(processed, (new_w, CONFIG.INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        padded = np.zeros((CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH), dtype=np.uint8)
        x_offset = (CONFIG.INPUT_WIDTH - new_w) // 2
        padded[:, x_offset:x_offset + new_w] = resized
        resized = padded
    
    if convert_to_rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    return resized, crop

class Vocabulary:
    """Vocabulary class for LaTeX tokenization."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in ['<pad>', '<start>', '<end>', '<unk>']:
            self.add_word(token)
        self.pad_token = self.word2idx['<pad>']
        self.start_token = self.word2idx['<start>']
        self.end_token = self.word2idx['<end>']
        self.unk_token = self.word2idx['<unk>']

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self) -> int:
        return len(self.word2idx)

    def tokenize(self, latex: str) -> List[int]:
        return [self.word2idx.get(char, self.unk_token) for char in latex.split()]

    def build_from_file(self, label_file: str):
        try:
            df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])
            tokens = sorted(set(' '.join(df['label'].astype(str)).split()))
            for token in tokens:
                self.add_word(token)
        except Exception as e:
            print(f"Error building vocabulary from {label_file}: {e}")

    def save(self, path: str):
        torch.save({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'idx': self.idx
        }, path)

    def load(self, path: str):
        data = torch.load(path)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx = data['idx']
        self._add_special_tokens()

class HMERDataset(Dataset):
    """Dataset for Handwritten Mathematical Expression Recognition."""
    def __init__(self, data_folder: str, label_file: str, vocab: Vocabulary, 
                 transform: Optional[A.Compose] = None, max_length: int = CONFIG.MAX_LENGTH):
        self.data_folder = data_folder
        self.max_length = max_length
        self.vocab = vocab
        self.transform = transform or A.Compose([
            A.Normalize(mean=[0.0], std=[1.0]),
            A.pytorch.ToTensorV2()
        ])

        # Load annotations
        df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])
        
        # Add file extension if needed
        if os.path.exists(data_folder):
            img_files = os.listdir(data_folder)
            if img_files:
                ext = os.path.splitext(img_files[0])[1]
                df['filename'] = df['filename'].apply(
                    lambda x: x if os.path.splitext(x)[1] else x + ext
                )

        self.annotations = dict(zip(df['filename'], df['label']))
        self.image_paths = list(self.annotations.keys())

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        latex = self.annotations[img_path]
        
        # Process image
        processed_img, _ = process_image(os.path.join(self.data_folder, img_path))
        processed_img = np.expand_dims(processed_img, axis=-1)
        
        # Apply transforms
        image = self.transform(image=processed_img)['image']
        
        # Tokenize
        tokens = [self.vocab.start_token] + self.vocab.tokenize(latex) + [self.vocab.end_token]
        tokens = tokens[:self.max_length]
        
        # Create count vector
        count_vector = torch.zeros(len(self.vocab))
        for token_id, count in Counter(tokens).items():
            if 0 <= token_id < len(count_vector):
                count_vector[token_id] = count
        
        # Pad tokens
        caption_length = torch.tensor([len(tokens)], dtype=torch.long)
        tokens = tokens + [self.vocab.pad_token] * (self.max_length - len(tokens))
        
        return image, torch.tensor(tokens, dtype=torch.long), caption_length, count_vector

def build_vocabulary(base_dir: str) -> Vocabulary:
    """Build unified vocabulary from all caption files."""
    vocab = Vocabulary()
    for subdir in [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]:
        caption_path = os.path.join(base_dir, subdir, 'caption.txt')
        if os.path.exists(caption_path):
            vocab.build_from_file(caption_path)
            print(f"Built vocabulary from {caption_path}")
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def create_dataloaders(base_dir: str = CONFIG.BASE_DIR) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """Create train, validation, and test dataloaders."""
    # Build vocabulary
    vocab = build_vocabulary(base_dir)
    os.makedirs('models', exist_ok=True)
    vocab.save('models/can/hmer_vocab.pth')

    # Define transform
    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),
        A.pytorch.ToTensorV2()
    ])

    # Create datasets
    train_dirs = ['train', '2014']
    train_datasets = []
    for train_dir in train_dirs:
        data_folder = os.path.join(base_dir, train_dir, 'img')
        label_file = os.path.join(base_dir, train_dir, 'caption.txt')
        if os.path.exists(data_folder) and os.path.exists(label_file):
            train_datasets.append(HMERDataset(data_folder, label_file, vocab, transform))

    if not train_datasets:
        raise ValueError("No training datasets found")
    
    # Validation dataset
    val_data_folder = os.path.join(base_dir, 'val', 'img')
    val_label_file = os.path.join(base_dir, 'val', 'caption.txt')
    if not (os.path.exists(val_data_folder) and os.path.exists(val_label_file)):
        val_data_folder = os.path.join(base_dir, '2016', 'img')
        val_label_file = os.path.join(base_dir, '2016', 'caption.txt')

    # Test dataset
    test_data_folder = os.path.join(base_dir, 'test', 'img')
    test_label_file = os.path.join(base_dir, 'test', 'caption.txt')
    if not (os.path.exists(test_data_folder) and os.path.exists(test_label_file)):
        test_data_folder = os.path.join(base_dir, '2019', 'img')
        test_label_file = os.path.join(base_dir, '2019', 'caption.txt')

    # Create dataloaders
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        HMERDataset(val_data_folder, val_label_file, vocab, transform),
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        HMERDataset(test_data_folder, test_label_file, vocab, transform),
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, vocab

def main():
    """Main function to demonstrate dataloader usage."""
    train_loader, val_loader, test_loader, vocab = create_dataloaders()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    for images, captions, lengths, count_vectors in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Caption batch shape: {captions.shape}")
        print(f"Lengths batch shape: {lengths.shape}")
        print(f"Count vectors batch shape: {count_vectors.shape}")
        break

# if __name__ == '__main__':
#     main()