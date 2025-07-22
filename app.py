import gradio as gr
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from models.can.can import CAN, create_can_model
from models.can.can_dataloader import Vocabulary, INPUT_HEIGHT, INPUT_WIDTH

# Load configuration
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)
CAN_CONFIG = cfg["can"]

# Global constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BACKBONE_TYPE = CAN_CONFIG["backbone_type"]
PRETRAINED_BACKBONE = True if CAN_CONFIG["pretrained_backbone"] == 1 else False
CHECKPOINT_PATH = f'checkpoints/{BACKBONE_TYPE}_can_best.pth' if not PRETRAINED_BACKBONE else f'checkpoints/p_{BACKBONE_TYPE}_can_best.pth'

# Modified process_img to accept numpy array and validate shapes
def process_img(image, convert_to_rgb=False):
    """
    Process a numpy array image: binarize, ensure black background, resize, and apply padding.

    Args:
        image: Numpy array (grayscale)
        convert_to_rgb: Whether to convert to RGB

    Returns:
        Processed image and crop information, or None if invalid
    """
    def is_effectively_binary(img, threshold_percentage=0.9):
        dark_pixels = np.sum(img < 20)
        bright_pixels = np.sum(img > 235)
        total_pixels = img.size
        return (dark_pixels + bright_pixels) / total_pixels > threshold_percentage

    def before_padding(image):
        if image.shape[0] < 2 or image.shape[1] < 2:
            return None, None  # Invalid image size
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:  # If image is normalized (0-1)
                image = (image * 255).astype(np.uint8)
            else:  # If image is in other float format
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        edges = cv2.Canny(image, 50, 150)
        kernel = np.ones((7, 13), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        sorted_components = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        best_f1 = 0
        best_crop = (0, 0, image.shape[1], image.shape[0])
        total_white_pixels = np.sum(dilated > 0)
        current_mask = np.zeros_like(dilated)
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0

        for component_idx in sorted_components:
            component_mask = labels == component_idx
            current_mask = np.logical_or(current_mask, component_mask)
            comp_y, comp_x = np.where(component_mask)
            if len(comp_x) > 0 and len(comp_y) > 0:
                x_min = min(x_min, np.min(comp_x))
                y_min = min(y_min, np.min(comp_y))
                x_max = max(x_max, np.max(comp_x))
                y_max = max(y_max, np.max(comp_y))
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            if width < 2 or height < 2:
                continue
            crop_area = width * height
            crop_mask = np.zeros_like(dilated)
            crop_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
            white_in_crop = np.sum(np.logical_and(dilated > 0, crop_mask > 0))
            precision = white_in_crop / crop_area if crop_area > 0 else 0
            recall = white_in_crop / total_white_pixels if total_white_pixels > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_crop = (x_min, y_min, x_max, y_max)

        x_min, y_min, x_max, y_max = best_crop
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        if cropped_image.shape[0] < 2 or cropped_image.shape[1] < 2:
            return None, None
        if is_effectively_binary(cropped_image):
            _, thresh = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
        else:
            thresh = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        white = np.sum(thresh == 255)
        black = np.sum(thresh == 0)
        if white > black:
            thresh = 255 - thresh
        denoised = cv2.medianBlur(thresh, 3)
        for _ in range(3):
            denoised = cv2.medianBlur(denoised, 3)
        result = cv2.copyMakeBorder(denoised, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        return result, best_crop

    if len(image.shape) != 2:
        return None, None  # Expect grayscale image
    
    # Ensure image is uint8 before processing
    if image.dtype != np.uint8:
        if image.max() <= 1.0:  # If image is normalized (0-1)
            image = (image * 255).astype(np.uint8)
        else:  # If image is in other float format
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    bin_img, best_crop = before_padding(image)
    if bin_img is None:
        return None, None
    h, w = bin_img.shape
    if h < 2 or w < 2:
        return None, None
    new_w = int((INPUT_HEIGHT / h) * w)

    if new_w > INPUT_WIDTH:
        resized_img = cv2.resize(bin_img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(bin_img, (new_w, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        padded_img = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
        x_offset = (INPUT_WIDTH - new_w) // 2
        padded_img[:, x_offset:x_offset + new_w] = resized_img
        resized_img = padded_img

    if convert_to_rgb:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    return resized_img, best_crop

# Load model and vocabulary
def load_checkpoint(checkpoint_path, device, pretrained_backbone=True, backbone='densenet'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint.get('vocab')
    if vocab is None:
        vocab_path = os.path.join(os.path.dirname(checkpoint_path), 'hmer_vocab.pth')
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path)
            vocab = Vocabulary()
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
            vocab.idx = vocab_data['idx']
            vocab.pad_token = vocab.word2idx['<pad>']
            vocab.start_token = vocab.word2idx['<start>']
            vocab.end_token = vocab.word2idx['<end>']
            vocab.unk_token = vocab.word2idx['<unk>']
        else:
            raise ValueError(f"Vocabulary not found in checkpoint and {vocab_path} does not exist")
    
    hidden_size = checkpoint.get('hidden_size', 256)
    embedding_dim = checkpoint.get('embedding_dim', 256)
    use_coverage = checkpoint.get('use_coverage', True)
    
    model = create_can_model(
        num_classes=len(vocab),
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        use_coverage=use_coverage,
        pretrained_backbone=pretrained_backbone,
        backbone_type=backbone
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, vocab

model, vocab = load_checkpoint(CHECKPOINT_PATH, DEVICE, PRETRAINED_BACKBONE, BACKBONE_TYPE)

# Image processing function for Gradio
def gradio_process_img(image, convert_to_rgb=False):
    # Convert Gradio image (PIL, numpy, or dict from Sketchpad) to grayscale numpy array
    if isinstance(image, dict):  # Handle Sketchpad input
        # The Sketchpad component returns a dict with 'background' and 'layers' keys
        # We need to combine the background and layers to get the final image
        background = np.array(image['background'])
        layers = image['layers']
        
        # Start with the background
        final_image = background.copy()
        
        # Add each layer on top
        for layer in layers:
            if layer is not None:  # Some layers might be None
                layer_img = np.array(layer)
                # Create a mask for non-transparent pixels
                mask = layer_img[..., 3] > 0
                # Replace pixels in final_image where mask is True, keeping the alpha channel
                final_image[mask] = layer_img[mask]
        
        # Convert to grayscale using the alpha channel
        if len(final_image.shape) == 3:
            # Use alpha channel to determine which pixels to keep
            alpha_mask = final_image[..., 3] > 0
            # Convert to grayscale using standard formula
            gray = np.dot(final_image[..., :3], [0.299, 0.587, 0.114])
            # Create a white background
            final_image = np.ones_like(gray) * 255
            # Apply the drawing where alpha > 0
            final_image[alpha_mask] = gray[alpha_mask]
            # Invert the image to get black on white
            final_image = 255 - final_image
    elif isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) != 2:
            raise ValueError("Invalid image format: Expected grayscale or RGB image")
    else:
        raise ValueError("Unsupported image input type")
    
    # For Sketchpad input, use the final_image we created
    if isinstance(image, dict):
        image = final_image
    
    # Apply modified process_img
    processed_img, best_crop = process_img(image, convert_to_rgb=False)
    if processed_img is None:
        raise ValueError("Image processing failed: Resulted in invalid image size")
    
    # Prepare for model input
    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),
        ToTensorV2()
    ])
    processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
    image_tensor = transform(image=processed_img)['image'].unsqueeze(0).to(DEVICE)
    
    return image_tensor, processed_img, best_crop

# Model inference
def recognize_image(image_tensor, processed_img, best_crop):
    with torch.no_grad():
        predictions, _ = model.recognize(
            image_tensor,
            max_length=150,
            start_token=vocab.start_token,
            end_token=vocab.end_token,
            beam_width=5
        )
    
    # Convert indices to LaTeX tokens
    latex_tokens = []
    for idx in predictions:
        if idx == vocab.end_token:
            break
        if idx != vocab.start_token:
            latex_tokens.append(vocab.idx2word[idx])
    
    latex = ' '.join(latex_tokens)
    
    # Format LaTeX for rendering
    rendered_latex = f"$${latex}$$"
    
    return latex, rendered_latex

# Gradio interface function
def process_draw(image):
    if image is None:
        return "Please draw an expression", ""
    
    try:
        # Process image
        image_tensor, processed_img, best_crop = gradio_process_img(image)
        
        # Recognize
        latex, rendered_latex = recognize_image(image_tensor, processed_img, best_crop)
        
        return latex, rendered_latex
    except Exception as e:
        return f"Error processing image: {str(e)}", ""

def process_upload(image):
    if image is None:
        return "Please upload an image", ""
    
    try:
        # Process image
        image_tensor, processed_img, best_crop = gradio_process_img(image)
        
        # Recognize
        latex, rendered_latex = recognize_image(image_tensor, processed_img, best_crop)
        
        return latex, rendered_latex
    except Exception as e:
        return f"Error processing image: {str(e)}", ""

# Enhanced custom CSS with expanded input areas
custom_css = """
/* Global styles */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', 'Roboto', sans-serif !important;
    padding: 1rem !important;
    box-sizing: border-box !important;
}

/* Header styling */
.header-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    font-size: clamp(1.8rem, 5vw, 2.5rem) !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    padding: 0 1rem !important;
}

/* Main container styling */
.main-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    border-radius: 20px !important;
    padding: clamp(1rem, 3vw, 2rem) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    margin: 1rem 0 !important;
}

/* Input section styling - RESPONSIVE */
.input-section {
    background: white !important;
    border-radius: 15px !important;
    padding: clamp(1rem, 3vw, 2rem) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    border: 1px solid #e1e8ed !important;
    min-height: min(700px, 80vh) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Output section styling - RESPONSIVE */
.output-section {
    background: white !important;
    border-radius: 15px !important;
    padding: clamp(1rem, 3vw, 1.5rem) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    border: 1px solid #e1e8ed !important;
    min-height: min(700px, 80vh) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Tab styling - RESPONSIVE */
.tab-nav {
    background: #f8f9fa !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
}

.tab-nav button {
    border-radius: 8px !important;
    padding: clamp(0.5rem, 2vw, 0.75rem) clamp(1rem, 3vw, 1.5rem) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border: none !important;
    background: transparent !important;
    color: #6c757d !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    white-space: nowrap !important;
}

/* Sketchpad styling - RESPONSIVE */
.sketchpad-container {
    border: 3px dashed #667eea !important;
    border-radius: 15px !important;
    background: #fafbfc !important;
    transition: all 0.3s ease !important;
    overflow: hidden !important;
    min-height: min(500px, 60vh) !important;
    height: min(500px, 60vh) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.sketchpad-container canvas {
    width: 100% !important;
    height: 100% !important;
    min-height: min(500px, 60vh) !important;
    touch-action: none !important;
}

/* Upload area styling - RESPONSIVE */
.upload-container {
    border: 3px dashed #667eea !important;
    border-radius: 15px !important;
    background: #fafbfc !important;
    padding: clamp(1.5rem, 5vw, 3rem) !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
    min-height: min(500px, 60vh) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    box-sizing: border-box !important;
}

.upload-container img {
    max-height: min(400px, 50vh) !important;
    max-width: 100% !important;
    object-fit: contain !important;
}

/* Button styling - RESPONSIVE */
.process-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: clamp(0.8rem, 2vw, 1.2rem) clamp(1.5rem, 4vw, 2.5rem) !important;
    font-size: clamp(1rem, 2.5vw, 1.2rem) !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
    white-space: nowrap !important;
}

/* Output text styling - RESPONSIVE */
.latex-output {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 10px !important;
    padding: clamp(1rem, 3vw, 1.5rem) !important;
    font-family: 'Monaco', 'Consolas', monospace !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    line-height: 1.6 !important;
    min-height: min(200px, 30vh) !important;
    overflow-x: auto !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

.rendered-output {
    background: white !important;
    border: 1px solid #e9ecef !important;
    border-radius: 10px !important;
    padding: clamp(1.5rem, 4vw, 2.5rem) !important;
    text-align: center !important;
    min-height: min(300px, 40vh) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: clamp(1.2rem, 3vw, 1.8rem) !important;
    overflow-x: auto !important;
}

/* Instructions styling - RESPONSIVE */
.instructions {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
    border-radius: 12px !important;
    padding: clamp(1rem, 3vw, 1.5rem) !important;
    margin-bottom: clamp(1rem, 3vw, 2rem) !important;
    border-left: 4px solid #28a745 !important;
}

.instructions h3 {
    color: #155724 !important;
    margin-bottom: 0.8rem !important;
    font-weight: 600 !important;
    font-size: clamp(1rem, 2.5vw, 1.1rem) !important;
}

.instructions p {
    color: #155724 !important;
    margin: 0.5rem 0 !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
    line-height: 1.5 !important;
}

/* Drawing tips styling - RESPONSIVE */
.drawing-tips {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
    border-radius: 10px !important;
    padding: clamp(0.8rem, 2vw, 1rem) !important;
    margin-top: 1rem !important;
    border-left: 4px solid #fd7e14 !important;
}

.drawing-tips h4 {
    color: #8a4100 !important;
    margin-bottom: 0.5rem !important;
    font-weight: 600 !important;
    font-size: clamp(0.9rem, 2vw, 1rem) !important;
}

.drawing-tips ul {
    color: #8a4100 !important;
    margin: 0 !important;
    padding-left: clamp(1rem, 3vw, 1.5rem) !important;
}

.drawing-tips li {
    margin: 0.3rem 0 !important;
    font-size: clamp(0.8rem, 1.8vw, 0.9rem) !important;
}

/* Full-width layout adjustments - RESPONSIVE */
.input-output-container {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 600px), 1fr)) !important;
    gap: clamp(1rem, 3vw, 2rem) !important;
    align-items: start !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Examples section - RESPONSIVE */
.examples-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)) !important;
    gap: clamp(1rem, 3vw, 1.5rem) !important;
    text-align: center !important;
}

.example-card {
    padding: clamp(1rem, 3vw, 1.5rem) !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    border-radius: 12px !important;
    border: 2px solid #dee2e6 !important;
}

.example-card strong {
    color: #495057 !important;
    font-size: clamp(0.9rem, 2.5vw, 1.1rem) !important;
    display: block !important;
    margin-bottom: 0.5rem !important;
}

.example-card span {
    font-family: monospace !important;
    color: #6c757d !important;
    font-size: clamp(0.8rem, 2vw, 0.9rem) !important;
    line-height: 1.6 !important;
}

/* Performance metrics section - RESPONSIVE */
.metrics-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)) !important;
    gap: clamp(0.8rem, 2vw, 1rem) !important;
}

.metric-item {
    text-align: center !important;
    padding: clamp(0.5rem, 2vw, 1rem) !important;
}

.metric-item strong {
    color: #e65100 !important;
    font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
    display: block !important;
    margin-bottom: 0.3rem !important;
}

.metric-item span {
    color: #bf360c !important;
    font-size: clamp(0.8rem, 2vw, 0.9rem) !important;
}

/* Responsive breakpoints */
@media (max-width: 1200px) {
    .gradio-container {
        padding: 0.8rem !important;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 0.5rem !important;
    }
    
    .main-container {
        padding: 0.8rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .input-section, .output-section {
        padding: 0.8rem !important;
    }
    
    .tab-nav {
        flex-direction: column !important;
    }
    
    .tab-nav button {
        width: 100% !important;
    }
}

@media (max-width: 480px) {
    .gradio-container {
        padding: 0.3rem !important;
    }
    
    .main-container {
        padding: 0.5rem !important;
        margin: 0.3rem 0 !important;
    }
    
    .input-section, .output-section {
        padding: 0.5rem !important;
    }
    
    .process-button {
        padding: 0.8rem 1.2rem !important;
        font-size: 0.9rem !important;
    }
}

/* Touch device optimizations */
@media (hover: none) {
    .process-button:hover {
        transform: none !important;
    }
    
    .sketchpad-container {
        touch-action: none !important;
        -webkit-touch-callout: none !important;
        -webkit-user-select: none !important;
        user-select: none !important;
    }
    
    .tab-nav button {
        padding: 1rem !important;
    }
}

/* Print styles */
@media print {
    .gradio-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    
    .input-section, .output-section {
        break-inside: avoid !important;
    }
    
    .process-button, .tab-nav {
        display: none !important;
    }
}
"""

# Create the enhanced Gradio interface with expanded input
with gr.Blocks(css=custom_css, title="Math Expression Recognition") as demo:
    gr.HTML('<h1 class="header-title">🧮 Handwritten Mathematical Expression Recognition</h1>')
    
    # Enhanced Instructions
    gr.HTML("""
    <div class="instructions">
        <h3>📝 How to use this expanded interface:</h3>
        <p><strong>✏️ Draw Tab:</strong> Use the large drawing canvas (800x500px) to draw mathematical expressions with your mouse or touch device</p>
        <p><strong>📁 Upload Tab:</strong> Upload high-resolution images containing handwritten mathematical expressions</p>
        <p><strong>🎯 Tips:</strong> Write clearly, use proper mathematical notation, and ensure good contrast between your writing and the background</p>
    </div>
    """)
    
    with gr.Row(elem_classes="input-output-container"):
        # Expanded Input Section
        with gr.Column(elem_classes="input-section"):
            gr.HTML('<h2 style="text-align: center; color: #667eea; margin-bottom: 1.5rem; font-size: 1.5rem;">📥 Input Area</h2>')
            
            with gr.Tabs():
                with gr.TabItem("✏️ Draw Expression"):
                    gr.HTML("""
                    <div class="drawing-tips">
                        <h4>🎨 Drawing Tips:</h4>
                        <ul>
                            <li>Use clear, legible handwriting</li>
                            <li>Draw symbols at reasonable sizes</li>
                            <li>Leave space between different parts</li>
                            <li>Use standard mathematical notation</li>
                            <li>Avoid overlapping symbols</li>
                        </ul>
                    </div>
                    """)
                    
                    # Add brush size instruction
                    gr.HTML("""
                    <div class="brush-instruction" style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; color: #1565c0;">
                        <strong>🖌️ Brush Size:</strong> Look for brush size controls in the drawing interface, or try different drawing pressures. 
                        For best results with math expressions, use a medium brush thickness.
                    </div>
                    """)
                    
                    draw_input = gr.Sketchpad(
                        label="Draw your mathematical expression here",
                        elem_classes="sketchpad-container",
                        height=500,
                        width=800,
                        canvas_size=(800, 500)
                    )
                    draw_button = gr.Button("🚀 Recognize Drawn Expression", elem_classes="process-button")
                
                with gr.TabItem("📁 Upload Image"):
                    gr.HTML("""
                    <div class="drawing-tips">
                        <h4>📷 Upload Tips:</h4>
                        <ul>
                            <li>Use high-resolution images (minimum 300 DPI)</li>
                            <li>Ensure good lighting and contrast</li>
                            <li>Crop the image to focus on the expression</li>
                            <li>Avoid shadows or glare</li>
                            <li>Supported formats: PNG, JPG, JPEG</li>
                        </ul>
                    </div>
                    """)
                    
                    upload_input = gr.Image(
                        label="Upload your mathematical expression image",
                        elem_classes="upload-container",
                        height=500,
                        type="pil"
                    )
                    upload_button = gr.Button("🚀 Recognize Uploaded Expression", elem_classes="process-button")
        
        # Output Section
        with gr.Column(elem_classes="output-section"):
            gr.HTML('<h2 style="text-align: center; color: #667eea; margin-bottom: 1.5rem; font-size: 1.5rem;">📤 Recognition Results</h2>')
            
            with gr.Tabs():
                with gr.TabItem("📄 LaTeX Code"):
                    latex_output = gr.Textbox(
                        label="Generated LaTeX Code",
                        elem_classes="latex-output",
                        lines=8,
                        placeholder="Your LaTeX code will appear here...\n\nThis is the raw LaTeX markup that represents your mathematical expression. You can copy this code and use it in any LaTeX document or LaTeX-compatible system.",
                        interactive=False
                    )
                
                with gr.TabItem("🎨 Rendered Expression"):
                    rendered_output = gr.Markdown(
                        label="Rendered Mathematical Expression",
                        elem_classes="rendered-output",
                        value="*Your beautifully rendered mathematical expression will appear here...*\n\n*Draw or upload an expression to see the magic happen!*"
                    )
    
    # Connect the buttons to their respective functions
    draw_button.click(
        fn=process_draw,
        inputs=[draw_input],
        outputs=[latex_output, rendered_output]
    )
    
    upload_button.click(
        fn=process_upload,
        inputs=[upload_input],
        outputs=[latex_output, rendered_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )