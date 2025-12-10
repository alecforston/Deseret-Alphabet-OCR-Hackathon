import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from model import CRNN
from dataset import DeseretTestDataset
from utils import load_vocab, decode_predictions, load_config


def predict(model, dataloader, device, idx_to_char):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for images, image_ids in tqdm(dataloader, desc="Generating predictions"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Decode predictions
            decoded = decode_predictions(outputs.permute(1, 0, 2), idx_to_char)
            
            # Store predictions
            for img_id, pred_text in zip(image_ids, decoded):
                predictions[img_id] = pred_text
    
    return predictions


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = Path(config['paths']['model_save_dir']) / 'vocab.json'
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Train the model first.")
    
    char_to_idx, idx_to_char = load_vocab(vocab_path)
    num_classes = len(char_to_idx)
    print(f"Loaded vocabulary with {num_classes} classes")
    
    # Create test dataset
    test_dataset = DeseretTestDataset(
        config['data']['test_images'],
        target_height=config['data']['target_height'],
        max_width=config['data']['max_width']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = CRNN(
        num_classes=num_classes,
        cnn_channels=config['model']['cnn_channels'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else \
                     Path(config['paths']['model_save_dir']) / 'crnn_best.pth'
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict(model, test_loader, device, idx_to_char)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame([
        {'id': img_id, 'label': pred_text}
        for img_id, pred_text in predictions.items()
    ])
    
    # Sort by ID
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # Save submission
    submission_dir = Path(config['paths']['submission_dir'])
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = args.output
    else:
        output_path = submission_dir / 'submission.csv'
    
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Show sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions for Deseret OCR')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: models/crnn_best.pth)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for submission CSV (default: submissions/submission.csv)')
    
    args = parser.parse_args()
    main(args)