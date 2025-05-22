import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from collections import Counter

import json 

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CAN_CONFIG = cfg["can"]


# Global constants
INPUT_HEIGHT = CAN_CONFIG["input_height"]
INPUT_WIDTH = CAN_CONFIG["input_width"]
BASE_DIR = CAN_CONFIG["base_dir"]
BATCH_SIZE = CAN_CONFIG["batch_size"]
NUM_WORKERS = CAN_CONFIG["num_workers"]


def is_effectively_binary(img, threshold_percentage=0.9):
    dark_pixels = np.sum(img < 20)
    bright_pixels = np.sum(img > 235)
    total_pixels = img.size

    return (dark_pixels + bright_pixels) / total_pixels > threshold_percentage


def before_padding(image):
    # Apply Canny edge detector to find text edges
    edges = cv2.Canny(image, 50, 150)

    # Apply dilation to connect nearby edges
    kernel = np.ones((7, 13), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    # Optimize crop rectangle using F1 score
    # Sort components by number of white pixels (excluding background which is label 0)
    sorted_components = sorted(
        range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True
    )

    # Initialize with empty crop
    best_f1 = 0
    best_crop = (0, 0, image.shape[1], image.shape[0])
    total_white_pixels = np.sum(dilated > 0)

    current_mask = np.zeros_like(dilated)
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0

    for component_idx in sorted_components:
        # Add this component to our mask
        component_mask = labels == component_idx
        current_mask = np.logical_or(current_mask, component_mask)

        # Update bounding box
        comp_y, comp_x = np.where(component_mask)
        if len(comp_x) > 0 and len(comp_y) > 0:
            x_min = min(x_min, np.min(comp_x))
            y_min = min(y_min, np.min(comp_y))
            x_max = max(x_max, np.max(comp_x))
            y_max = max(y_max, np.max(comp_y))

        # Calculate the current crop
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        crop_area = width * height

        crop_mask = np.zeros_like(dilated)
        crop_mask[y_min : y_max + 1, x_min : x_max + 1] = 1
        white_in_crop = np.sum(np.logical_and(dilated > 0, crop_mask > 0))

        # Calculate F1 score
        precision = white_in_crop / crop_area
        recall = white_in_crop / total_white_pixels
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_crop = (x_min, y_min, x_max, y_max)

    # Apply the best crop to the original image
    x_min, y_min, x_max, y_max = best_crop
    cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]

    # Apply Gaussian adaptive thresholding
    if is_effectively_binary(cropped_image):
        _, thresh = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # Ensure background is black
    white = np.sum(thresh == 255)
    black = np.sum(thresh == 0)
    if white > black:
        thresh = 255 - thresh

    # Clean up noise using median filter
    denoised = cv2.medianBlur(thresh, 3)
    for _ in range(3):
        denoised = cv2.medianBlur(denoised, 3)

    # Add padding
    result = cv2.copyMakeBorder(denoised, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)

    return result, best_crop


def process_img(filename, convert_to_rgb=False):
    """
    Load, binarize, ensure black background, resize, and apply padding

    Args:
        filename: Path to the image file
        convert_to_rgb: Whether to convert to RGB

    Returns:
        Processed image and crop information
    """
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image file: {filename}")

    bin_img, best_crop = before_padding(image)
    h, w = bin_img.shape
    new_w = int((INPUT_HEIGHT / h) * w)

    if new_w > INPUT_WIDTH:
        resized_img = cv2.resize(
            bin_img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
        )
    else:
        resized_img = cv2.resize(
            bin_img, (new_w, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
        )
        padded_img = (
            np.ones((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8) * 0
        )  # Black background
        x_offset = (INPUT_WIDTH - new_w) // 2
        padded_img[:, x_offset : x_offset + new_w] = resized_img
        resized_img = padded_img

    # Convert to BGR/RGB only if necessary
    if convert_to_rgb:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    return resized_img, best_crop


class HMERDatasetForCAN(Dataset):
    """
    Dataset integrated with the CAN model for HMER
    """

    def __init__(self, data_folder, label_file, vocab, transform=None, max_length=150):
        """
        Initialize the dataset

        data_folder: Directory containing images
        label_file: TSV file with two columns (filename, label), no header
        vocab: Vocabulary object for tokenization
        transform: Image transformations
        max_length: Maximum length of the token sequence
        """
        self.data_folder = data_folder
        self.max_length = max_length
        self.vocab = vocab

        # Read the label file
        df = pd.read_csv(label_file, sep="\t", header=None, names=["filename", "label"])

        # Check image file format
        if os.path.exists(data_folder):
            img_files = os.listdir(data_folder)
            if img_files:
                # Get the extension of the first file
                extension = os.path.splitext(img_files[0])[1]
                # Add extension to filenames if not present
                df["filename"] = df["filename"].apply(
                    lambda x: x if os.path.splitext(x)[1] else x + extension
                )

        self.annotations = dict(zip(df["filename"], df["label"]))
        self.image_paths = list(self.annotations.keys())

        # Default transformation
        if transform is None:
            transform = A.Compose(
                [
                    A.Normalize(
                        mean=[0.0], std=[1.0]
                    ),  # Normalize for single channel (grayscale)   
                    A.pytorch.ToTensorV2(),
                ]
            )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path and LaTeX expression
        image_path = self.image_paths[idx]
        latex = self.annotations[image_path]

        # Process image
        file_path = os.path.join(self.data_folder, image_path)
        processed_img, _ = process_img(
            file_path, convert_to_rgb=False
        )  # Keep image as grayscale

        # Convert to [C, H, W] format and normalize
        if self.transform:
            # Ensure image has the correct format for albumentations
            processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
            image = self.transform(image=processed_img)["image"]
        else:
            # If no transform, manually convert to tensor
            image = torch.from_numpy(processed_img).float() / 255.0
            image = image.unsqueeze(0)  # Add grayscale channel: [1, H, W]

        # Tokenize LaTeX expression
        tokens = self.vocab.tokenize(latex)

        # Add start and end tokens
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        # Truncate if exceeding max length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        # Create counting vector for CAN
        count_vector = self.create_count_vector(tokens)

        # Store actual caption length
        caption_length = torch.LongTensor([len(tokens)])

        # Pad to max length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab.pad_token] * (self.max_length - len(tokens))

        # Convert to tensor
        caption = torch.LongTensor(tokens)

        return image, caption, caption_length, count_vector

    def create_count_vector(self, tokens):
        """
        Create counting vector for the CAN model

        Args:
            tokens: List of token IDs

        Returns:
            Tensor counting the occurrence of each symbol
        """
        # Count occurrences of each token
        counter = Counter(tokens)

        # Create counting vector with size equal to vocabulary size
        count_vector = torch.zeros(len(self.vocab))

        # Fill counting vector with counts
        for token_id, count in counter.items():
            if 0 <= token_id < len(count_vector):
                count_vector[token_id] = count

        return count_vector


class Vocabulary:
    """
    Advanced Vocabulary class for tokenization
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add special tokens
        self.add_word("<pad>")  # Padding token
        self.add_word("<start>")  # Start token
        self.add_word("<end>")  # End token
        self.add_word("<unk>")  # Unknown token

        self.pad_token = self.word2idx["<pad>"]
        self.start_token = self.word2idx["<start>"]
        self.end_token = self.word2idx["<end>"]
        self.unk_token = self.word2idx["<unk>"]

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, latex):
        """
        Tokenize LaTeX string into indices. Assumes tokens are space-separated.
        """
        tokens = []

        for char in latex.split():
            if char in self.word2idx:
                tokens.append(self.word2idx[char])
            else:
                tokens.append(self.unk_token)

        return tokens

    def build_vocab(self, label_file):
        """
        Build vocabulary from label file
        """
        try:
            df = pd.read_csv(
                label_file, sep="\t", header=None, names=["filename", "label"]
            )
            all_labels_text = " ".join(df["label"].astype(str).tolist())
            tokens = sorted(set(all_labels_text.split()))
            for char in tokens:
                self.add_word(char)
        except Exception as e:
            print(f"Error building vocabulary from {label_file}: {e}")

    def save_vocab(self, path):
        """
        Save vocabulary to file
        """
        data = {"word2idx": self.word2idx, "idx2word": self.idx2word, "idx": self.idx}
        torch.save(data, path)

    def load_vocab(self, path):
        """
        Load vocabulary from file
        """
        data = torch.load(path)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self.idx = data["idx"]

        # Update special tokens
        self.pad_token = self.word2idx["<pad>"]
        self.start_token = self.word2idx["<start>"]
        self.end_token = self.word2idx["<end>"]
        self.unk_token = self.word2idx["<unk>"]


def build_unified_vocabulary(base_dir="data/CROHME"):
    """
    Build a unified vocabulary from all caption.txt files

    Args:
        base_dir: Root directory containing CROHME data

    Returns:
        Constructed Vocabulary object
    """
    vocab = Vocabulary()
    # Get all subdirectories
    subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for subdir in subdirs:
        caption_path = os.path.join(base_dir, subdir, "caption.txt")
        if os.path.exists(caption_path):
            vocab.build_vocab(caption_path)
            print(f"Built vocabulary from {caption_path}")

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab


def create_dataloaders_for_can(base_dir="data/CROHME", batch_size=32, num_workers=4):
    """
    Create dataloaders for training the CAN model

    Args:
        base_dir: Root directory containing CROHME data
        batch_size: Batch size
        num_workers: Number of workers for DataLoader

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    # Build unified vocabulary
    vocab = build_unified_vocabulary(base_dir)

    # Save vocabulary for later use
    os.makedirs("models", exist_ok=True)
    vocab.save_vocab("models/hmer_vocab.pth")

    # Create transform for grayscale data
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0], std=[1.0]
            ),  # Normalize for single channel (grayscale)   
            A.pytorch.ToTensorV2(),
        ]
    )

    # Create datasets
    train_datasets = []

    # Use 'train' and possibly add other datasets to training set
    train_dirs = ["train", "2014"]  # Add other directories if desired
    for train_dir in train_dirs:
        data_folder = os.path.join(base_dir, train_dir, "img")
        label_file = os.path.join(base_dir, train_dir, "caption.txt")

        if os.path.exists(data_folder) and os.path.exists(label_file):
            train_datasets.append(
                HMERDatasetForCAN(
                    data_folder=data_folder,
                    label_file=label_file,
                    vocab=vocab,
                    transform=transform,
                )
            )

    # Combine training datasets
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
    else:
        raise ValueError("No training datasets found")

    # Validation dataset
    val_data_folder = os.path.join(base_dir, "val", "img")
    val_label_file = os.path.join(base_dir, "val", "caption.txt")

    if not os.path.exists(val_data_folder) or not os.path.exists(val_label_file):
        # Use '2016' as validation set if 'val' is not available
        val_data_folder = os.path.join(base_dir, "2016", "img")
        val_label_file = os.path.join(base_dir, "2016", "caption.txt")

    val_dataset = HMERDatasetForCAN(
        data_folder=val_data_folder,
        label_file=val_label_file,
        vocab=vocab,
        transform=transform,
    )

    # Test dataset
    test_data_folder = os.path.join(base_dir, "test", "img")
    test_label_file = os.path.join(base_dir, "test", "caption.txt")

    if not os.path.exists(test_data_folder) or not os.path.exists(test_label_file):
        # Use '2019' as test set if 'test' is not available
        test_data_folder = os.path.join(base_dir, "2019", "img")
        test_label_file = os.path.join(base_dir, "2019", "caption.txt")

    test_dataset = HMERDatasetForCAN(
        data_folder=test_data_folder,
        label_file=test_label_file,
        vocab=vocab,
        transform=transform,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab


# Use functionality integrated with the CAN model
def main():
    # Create dataloader for the CAN model
    train_loader, val_loader, test_loader, vocab = create_dataloaders_for_can(
        base_dir=BASE_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Print information
    print(f"Training samples: {len(train_loader.dataset)}")  
    print(f"Validation samples: {len(val_loader.dataset)}")  
    print(f"Test samples: {len(test_loader.dataset)}")  

    # Check dataloader output
    for images, captions, lengths, count_vectors in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Caption batch shape: {captions.shape}")
        print(f"Lengths batch shape: {lengths.shape}")
        print(f"Count vectors batch shape: {count_vectors.shape}")
        break


if __name__ == "__main__":
    main()
