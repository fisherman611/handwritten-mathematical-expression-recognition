import os
import sys

import torch
import pandas as pd
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import json
import torch.nn.functional as F

from can import CAN, create_can_model
from can_dataloader import Vocabulary, process_img, INPUT_HEIGHT, INPUT_WIDTH

torch.serialization.add_safe_globals([Vocabulary])

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CAN_CONFIG = cfg["can"]


# Global constants here
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODE = CAN_CONFIG["mode"]  # 'single' or 'evaluate'
BACKBONE_TYPE = CAN_CONFIG["backbone_type"]
PRETRAINED_BACKBONE = True if CAN_CONFIG["pretrained_backbone"] == 1 else False
CHECKPOINT_PATH = f'checkpoints/{BACKBONE_TYPE}_can_best.pth' if PRETRAINED_BACKBONE == False else f'checkpoints/p_{BACKBONE_TYPE}_can_best.pth'
IMAGE_PATH = f"{CAN_CONFIG["test_folder"]}/{CAN_CONFIG["relative_image_path"]}"
VISUALIZE = True if CAN_CONFIG["visualize"] == 1 else False
TEST_FOLDER = CAN_CONFIG["test_folder"]
LABEL_FILE = CAN_CONFIG["label_file"]
CLASSIFIER = CAN_CONFIG["classifier"]  # choose between 'frac', 'sum_or_lim', 'long_expr', and 'all'


def filter_formula(formula_tokens, mode):
    if mode == "frac":
        return "\\frac" in formula_tokens
    elif mode == "sum_or_lim":
        return "\\sum" in formula_tokens or "\\limit" in formula_tokens
    elif mode == "long_expr":
        return len(formula_tokens) >= 30
    elif mode == 'short_expr':
        return len(formula_tokens) <= 10
    return True


def levenshtein_distance(lst1, lst2):
    """
    Calculate Levenshtein distance between two lists
    """
    m = len(lst1)
    n = len(lst2)

    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)
    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            if lst1[i - 1] == lst2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    curr_row[j - 1],  # insertion
                    prev_row[j],  # deletion
                    prev_row[j - 1]  # substitution
                )

        prev_row = curr_row.copy()
    return curr_row[n]


def load_checkpoint(checkpoint_path, device, pretrained_backbone=True, backbone='densenet'):
    """
    Load checkpoint and return model and vocabulary
    """
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=False)

    vocab = checkpoint.get('vocab')
    if vocab is None:
        # Try to load vocab from a separate file if not in checkpoint
        vocab_path = os.path.join(os.path.dirname(checkpoint_path),
                                  'hmer_vocab.pth')
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path)
            vocab = Vocabulary()
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
            vocab.idx = vocab_data['idx']
            # Update special tokens
            vocab.pad_token = vocab.word2idx['<pad>']
            vocab.start_token = vocab.word2idx['<start>']
            vocab.end_token = vocab.word2idx['<end>']
            vocab.unk_token = vocab.word2idx['<unk>']
        else:
            raise ValueError(
                f"Vocabulary not found in checkpoint and {vocab_path} does not exist"
            )

    # Initialize model with parameters from checkpoint
    hidden_size = checkpoint.get('hidden_size', 256)
    embedding_dim = checkpoint.get('embedding_dim', 256)
    use_coverage = checkpoint.get('use_coverage', True)

    model = create_can_model(num_classes=len(vocab),
                             hidden_size=hidden_size,
                             embedding_dim=embedding_dim,
                             use_coverage=use_coverage,
                             pretrained_backbone=pretrained_backbone,
                             backbone_type=backbone).to(device)

    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model from checkpoint {checkpoint_path}")

    return model, vocab


def recognize_single_image(model,
                           image_path,
                           vocab,
                           device,
                           max_length=150,
                           visualize_attention=False):
    """
    Recognize handwritten mathematical expression from a single image using the CAN model
    """
    # Prepare image transform for grayscale images
    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),  # For grayscale        
        A.pytorch.ToTensorV2()
    ])

    # Load and transform image
    processed_img, best_crop = process_img(image_path, convert_to_rgb=False)

    # Ensure image has the correct format for albumentations
    processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
    image_tensor = transform(
        image=processed_img)['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Generate LaTeX using beam search
        predictions, attention_weights = model.recognize(
            image_tensor,
            max_length=max_length,
            start_token=vocab.start_token,
            end_token=vocab.end_token,
            beam_width=5  # Use beam search with width 5
        )

    # Convert indices to LaTeX tokens
    latex_tokens = []
    for idx in predictions:
        if idx == vocab.end_token:
            break
        if idx != vocab.start_token:  # Skip start token
            latex_tokens.append(vocab.idx2word[idx])

    # Join tokens to get complete LaTeX
    latex = ' '.join(latex_tokens)

    # Visualize attention if requested
    if visualize_attention and attention_weights is not None:
        visualize_attention_maps(processed_img, attention_weights,
                                 latex_tokens, best_crop)

    return latex


def visualize_attention_maps(orig_image,
                             attention_weights,
                             latex_tokens,
                             best_crop,
                             max_cols=4):
    """
    Visualize attention maps over the image for CAN model
    """
    # Create PIL image from numpy array
    orig_image = orig_image.crop(best_crop)
    orig_w, orig_h = orig_image.size
    ratio = INPUT_HEIGHT / INPUT_WIDTH
    
    num_tokens = len(latex_tokens)
    num_cols = min(max_cols, num_tokens)
    num_rows = int(np.ceil(num_tokens / num_cols))

    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=(num_cols * 3, int(num_rows * 6 * orig_h / orig_w)))
    axes = np.array(axes).reshape(-1)

    for i, (token, attn) in enumerate(zip(latex_tokens, attention_weights)):
        ax = axes[i]

        attn = attn[0:1].squeeze(0)
        attn_len = attn.shape[0]
        attn_w = int(np.sqrt(attn_len / ratio))
        attn_h = int(np.sqrt(attn_len * ratio))

        # resize to (orig_h, interpolated_w)
        attn = attn.view(1, 1, attn_h, attn_w)
        interp_w = int(orig_h / ratio)

        attn = F.interpolate(attn, size=(orig_h, interp_w), mode='bilinear', align_corners=False)
        attn = attn.squeeze().cpu().numpy()

        # fix aspect ratio mismatch
        if interp_w > orig_w:
            # center crop width
            start = (interp_w - orig_w) // 2
            attn = attn[:, start:start + orig_w]
        elif interp_w < orig_w:
            # stretch to fit width
            attn = cv2.resize(attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        ax.imshow(orig_image)
        ax.imshow(attn, cmap='jet', alpha=0.4)
        ax.set_title(f'{token}', fontsize=10 * 8 * orig_h / orig_w)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('attention_maps_can.png', bbox_inches='tight', dpi=150)
    plt.close()


def evaluate_model(model,
                   test_folder,
                   label_file,
                   vocab,
                   device,
                   max_length=150,
                   batch_size=32):
    """
    Evaluate CAN model on test set
    """
    df = pd.read_csv(label_file,
                     sep='\t',
                     header=None,
                     names=['filename', 'label'])

    # Check image file format
    if os.path.exists(test_folder):
        img_files = os.listdir(test_folder)
        if img_files:
            # Get the extension of the first file
            extension = os.path.splitext(img_files[0])[1]
            # Add extension to filenames if not present
            df['filename'] = df['filename'].apply(
                lambda x: x if os.path.splitext(x)[1] else x + extension)

    annotations = dict(zip(df['filename'], df['label']))

    model.eval()

    correct = 0
    err1 = 0
    err2 = 0
    err3 = 0
    total = 0

    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),  # For grayscale            
        A.pytorch.ToTensorV2()
    ])

    results = {}

    for image_path, gt_latex in tqdm(annotations.items(), desc="Evaluating"):
        gt_latex: str = gt_latex
        if not filter_formula(gt_latex.split(), CLASSIFIER):
            continue 
        file_path = os.path.join(test_folder, image_path)

        try:
            processed_img, _ = process_img(file_path, convert_to_rgb=False)

            # Ensure image has the correct format for albumentations
            processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
            image_tensor = transform(
                image=processed_img)['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                predictions, _ = model.recognize(
                    image_tensor,
                    max_length=max_length,
                    start_token=vocab.start_token,
                    end_token=vocab.end_token,
                    beam_width=5  # Use beam search
                )

            # Convert indices to LaTeX tokens
            pred_latex_tokens = []
            for idx in predictions:
                if idx == vocab.end_token:
                    break
                if idx != vocab.start_token:  # Skip start token
                    pred_latex_tokens.append(vocab.idx2word[idx])

            pred_latex = ' '.join(pred_latex_tokens)

            gt_latex_tokens = gt_latex.split()
            edit_distance = levenshtein_distance(pred_latex_tokens,
                                                 gt_latex_tokens)

            if edit_distance == 0:
                correct += 1
            elif edit_distance == 1:
                err1 += 1
            elif edit_distance == 2:
                err2 += 1
            elif edit_distance == 3:
                err3 += 1

            total += 1

            # Save result
            results[image_path] = {
                'ground_truth': gt_latex,
                'prediction': pred_latex,
                'edit_distance': edit_distance
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Calculate accuracy metrics
    exprate = round(correct / total, 4) if total > 0 else 0
    exprate_leq1 = round((correct + err1) / total, 4) if total > 0 else 0
    exprate_leq2 = round(
        (correct + err1 + err2) / total, 4) if total > 0 else 0
    exprate_leq3 = round(
        (correct + err1 + err2 + err3) / total, 4) if total > 0 else 0

    print(f"Exact match rate: {exprate:.4f}")
    print(f"Edit distance ≤ 1: {exprate_leq1:.4f}")
    print(f"Edit distance ≤ 2: {exprate_leq2:.4f}")
    print(f"Edit distance ≤ 3: {exprate_leq3:.4f}")

    # Save results to file
    with open('evaluation_results_can.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'accuracy': {
                    'exprate': exprate,
                    'exprate_leq1': exprate_leq1,
                    'exprate_leq2': exprate_leq2,
                    'exprate_leq3': exprate_leq3
                },
                'results': results
            },
            f,
            indent=4)

    return {
        'exprate': exprate,
        'exprate_leq1': exprate_leq1,
        'exprate_leq2': exprate_leq2,
        'exprate_leq3': exprate_leq3
    }, results


def main(mode):
    device = DEVICE
    print(f'Using device: {device}')

    checkpoint_path = CHECKPOINT_PATH
    backbone = BACKBONE_TYPE
    pretrained_backbone = PRETRAINED_BACKBONE

    # For single mode
    image_path = IMAGE_PATH
    visualize = VISUALIZE

    # For evaluation mode
    test_folder = TEST_FOLDER
    label_file = LABEL_FILE

    # Load model and vocabulary
    model, vocab = load_checkpoint(checkpoint_path, device, pretrained_backbone=pretrained_backbone, backbone=backbone)

    if mode == 'single':
        if image_path is None:
            raise ValueError('Image path is required for single mode')

        latex = recognize_single_image(model,
                                       image_path,
                                       vocab,
                                       device,
                                       visualize_attention=visualize)
        print(f'Recognized LaTeX: {latex}')

    elif mode == 'evaluate':
        if test_folder is None or label_file is None:
            raise ValueError(
                'Test folder and annotation file are required for evaluate mode'
            )

        metrics, results = evaluate_model(model, test_folder, label_file,
                                          vocab, device)
        print(f"##### Score of {CLASSIFIER} expression type: #####")
        print(f'Evaluation metrics: {metrics}')


if __name__ == '__main__':
    # Ensure Vocabulary is safe for serialization
    torch.serialization.add_safe_globals([Vocabulary])

    # Run the main function
    main(MODE)