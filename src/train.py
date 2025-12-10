import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
from tqdm import tqdm

from model import CRNN, count_parameters
from dataset import DeseretDataset, collate_fn
from utils import (set_seed, load_config, build_vocab, save_vocab, 
                   decode_predictions, calculate_levenshtein,
                   AverageMeter, save_checkpoint, load_checkpoint)

def train_epoch(model, dataloader, criterion, optimizer, device, idx_to_char, epoch):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    lev_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels, label_lengths) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)  # (seq_len, batch, num_classes)
        
        # Calculate input lengths (sequence length for each sample)
        seq_len = outputs.size(0)
        input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long)
        
        # CTC Loss
        log_probs = nn.functional.log_softmax(outputs, dim=2)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = decode_predictions(outputs.permute(1, 0, 2), idx_to_char)
            
            # Get target texts
            targets = []
            offset = 0
            for length in label_lengths:
                target_indices = labels[offset:offset+length].cpu().numpy()
                target_text = ''.join([idx_to_char[idx] for idx in target_indices if idx in idx_to_char])
                targets.append(target_text)
                offset += length
            
            lev_dist = calculate_levenshtein(predictions, targets)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        lev_meter.update(lev_dist, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lev': f'{lev_meter.avg:.2f}'
        })
    
    return loss_meter.avg, lev_meter.avg


def validate(model, dataloader, criterion, device, idx_to_char):
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    lev_meter = AverageMeter()
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate input lengths
            seq_len = outputs.size(0)
            input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long)
            
            # CTC Loss
            log_probs = nn.functional.log_softmax(outputs, dim=2)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            # Decode predictions
            predictions = decode_predictions(outputs.permute(1, 0, 2), idx_to_char)
            
            # Get target texts
            targets = []
            offset = 0
            for length in label_lengths:
                target_indices = labels[offset:offset+length].cpu().numpy()
                target_text = ''.join([idx_to_char[idx] for idx in target_indices if idx in idx_to_char])
                targets.append(target_text)
                offset += length
            
            lev_dist = calculate_levenshtein(predictions, targets)
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            lev_meter.update(lev_dist, images.size(0))
    
    return loss_meter.avg, lev_meter.avg


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build or load vocabulary
    vocab_path = Path(config['paths']['model_save_dir']) / 'vocab.json'
    if vocab_path.exists() and not args.rebuild_vocab:
        print(f"Loading vocabulary from {vocab_path}")
        from utils import load_vocab
        char_to_idx, idx_to_char = load_vocab(vocab_path)
    else:
        print("Building vocabulary from training data...")
        char_to_idx, idx_to_char = build_vocab(config['data']['train_labels'])
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        save_vocab(char_to_idx, vocab_path)
    
    num_classes = len(char_to_idx)
    print(f"Number of classes: {num_classes}")
    
    # Create datasets
    full_dataset = DeseretDataset(
        config['data']['train_images'],
        config['data']['train_labels'],
        char_to_idx,
        target_height=config['data']['target_height'],
        max_width=config['data']['max_width'],
        augment=True
    )
    
    # Split into train and validation
    train_size = int(config['data']['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Turn off augmentation for validation
    val_dataset.dataset.augment = False
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    # Set pin_memory based on device (only useful for CUDA)
    use_pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    # Create model
    model = CRNN(
        num_classes=num_classes,
        cnn_channels=config['model']['cnn_channels'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function (CTC Loss)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler (fixed - removed 'verbose' parameter)
    scheduler = None
    if config['training']['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler_factor'],
            patience=config['training']['scheduler_patience']
        )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    # Training loop
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_lev = train_epoch(
            model, train_loader, criterion, optimizer, device, idx_to_char, epoch + 1
        )
        
        # Validate
        val_loss, val_lev = validate(model, val_loader, criterion, device, idx_to_char)
        
        print(f"Train Loss: {train_loss:.4f}, Train Lev: {train_lev:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val Lev: {val_lev:.2f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_path = Path(config['paths']['model_save_dir']) / 'crnn_best.pth'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'config': config
            }, save_path)
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")

        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deseret CRNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--rebuild_vocab', action='store_true',
                       help='Rebuild vocabulary even if it exists')
    
    args = parser.parse_args()
    main(args)