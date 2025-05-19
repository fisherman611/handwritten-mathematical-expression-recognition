import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import time
import wandb
from datetime import datetime
from tqdm.auto import tqdm

# Import model and data loader from previous files
from models.can.can import CAN
from models.can.can_dataloader import create_dataloaders_for_can, Vocabulary

import albumentations as A
import cv2
import random

class RandomMorphology(A.ImageOnlyTransform):
    def __init__(self, p=0.5, kernel_size=3):
        super(RandomMorphology, self).__init__(p)
        self.kernel_size = kernel_size

    def apply(self, img, **params):
        op = random.choice(['erode', 'dilate'])
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        if op == 'erode':
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)

# Custom transforms for CAN model (grayscale images)
train_transforms = A.Compose([
    A.Rotate(limit=5, p=0.25, border_mode=cv2.BORDER_REPLICATE),
    A.ElasticTransform(alpha=100, sigma=7, p=0.5, interpolation=cv2.INTER_CUBIC),
    RandomMorphology(p=0.5, kernel_size=2),
    A.Normalize(mean=[0.0], std=[1.0]),  # For grayscale      #type: ignore
    A.pytorch.ToTensorV2()
])

def train_epoch(model, train_loader, optimizer, device, grad_clip=5.0, lambda_count=0.01, print_freq=10):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_count_loss = 0.0
    batch_count = 0

    for i, (images, captions, caption_lengths, count_targets) in tqdm(enumerate(train_loader)):
        batch_count += 1
        images = images.to(device)
        captions = captions.to(device)
        count_targets = count_targets.to(device)

        # Forward pass
        outputs, count_vectors = model(images, captions, teacher_forcing_ratio=0.5)
        
        # Calculate loss
        loss, cls_loss, counting_loss = model.calculate_loss(
            outputs=outputs, 
            targets=captions, 
            count_vectors=count_vectors, 
            count_targets=count_targets,
            lambda_count=lambda_count
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_count_loss += counting_loss.item()

        # Print progress
        if i % print_freq == 0 and i > 0:
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                  f'Cls Loss: {cls_loss.item():.4f}, Count Loss: {counting_loss.item():.4f}')

    return total_loss / batch_count, total_cls_loss / batch_count, total_count_loss / batch_count

def validate(model, val_loader, device, lambda_count=0.01):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_count_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for i, (images, captions, caption_lengths, count_targets) in tqdm(enumerate(val_loader)):
            batch_count += 1
            images = images.to(device)
            captions = captions.to(device)
            count_targets = count_targets.to(device)

            # Forward pass
            outputs, count_vectors = model(images, captions, teacher_forcing_ratio=0.0)  # No teacher forcing in validation
            
            # Calculate loss
            loss, cls_loss, counting_loss = model.calculate_loss(
                outputs=outputs, 
                targets=captions, 
                count_vectors=count_vectors, 
                count_targets=count_targets,
                lambda_count=lambda_count
            )

            # Track losses
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_count_loss += counting_loss.item()

    return total_loss / batch_count, total_cls_loss / batch_count, total_count_loss / batch_count

def main():
    # Configuration
    dataset_dir = 'data/CROHME'
    seed = 1337
    checkpoints_dir = 'checkpoints'
    batch_size = 16
    
    # Model parameters
    hidden_size = 256
    embedding_dim = 256
    use_coverage = True
    lambda_count = 0.01
    
    # Training parameters
    lr = 3e-4
    epochs = 100
    grad_clip = 5.0
    print_freq = 20
    
    # Scheduler parameters
    T_0 = 5
    T_mult = 2
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create checkpoint directory
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders_for_can(
        base_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")       #type: ignore
    print(f"Validation samples: {len(val_loader.dataset)}")       #type: ignore
    print(f"Test samples: {len(test_loader.dataset)}")            #type: ignore
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    model = CAN(
        num_classes=len(vocab),
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        use_coverage=use_coverage
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult
    )
    
    # Initialize wandb
    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb.init(project='hmer-can', name=run_name, config={
        'seed': seed,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'embedding_dim': embedding_dim,
        'use_coverage': use_coverage,
        'lambda_count': lambda_count,
        'lr': lr,
        'epochs': epochs,
        'grad_clip': grad_clip,
        'T_0': T_0,
        'T_mult': T_mult
    })
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        curr_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1:03}/{epochs:03}')
        t1 = time.time()
        
        # Train
        train_loss, train_cls_loss, train_count_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            lambda_count=lambda_count,
            print_freq=print_freq
        )
        
        # Validate
        val_loss, val_cls_loss, val_count_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            lambda_count=lambda_count
        )
        
        # Update learning rate
        scheduler.step()
        t2 = time.time()
        
        # Print stats
        print(f'Train - Total Loss: {train_loss:.4f}, Class Loss: {train_cls_loss:.4f}, Count Loss: {train_count_loss:.4f}')
        print(f'Val - Total Loss: {val_loss:.4f}, Class Loss: {val_cls_loss:.4f}, Count Loss: {val_count_loss:.4f}')
        print(f'Time: {t2 - t1:.2f}s, Learning Rate: {curr_lr:.6f}')
        
        # Log metrics to wandb
        wandb.log({
            'train_loss': train_loss,
            'train_cls_loss': train_cls_loss,
            'train_count_loss': train_count_loss,
            'val_loss': val_loss,
            'val_cls_loss': val_cls_loss,
            'val_count_loss': val_count_loss,
            'learning_rate': curr_lr,
            'epoch': epoch
        })
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab
            }
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'can_best.pth'))
            print('Model saved!')
    
    print('Training completed!')

if __name__ == '__main__':
    main()