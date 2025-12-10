import json
import torch
import random
import numpy as np
from pathlib import Path
import yaml
import editdistance


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_vocab(label_dir):
    """
    Build character vocabulary from training labels
    
    Returns:
        char_to_idx: dict mapping characters to indices
        idx_to_char: dict mapping indices to characters
    """
    label_path = Path(label_dir)
    all_chars = set()
    
    for label_file in label_path.glob('*.txt'):
        with open(label_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            all_chars.update(text)
    
    # Sort for consistency
    chars = sorted(list(all_chars))
    
    # Reserve 0 for CTC blank
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx['<blank>'] = 0
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(char_to_idx)} characters")
    print(f"Characters: {chars}")
    
    return char_to_idx, idx_to_char


def save_vocab(char_to_idx, save_path):
    """Save vocabulary to JSON file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path):
    """Load vocabulary from JSON file"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)
    idx_to_char = {int(idx): char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def decode_predictions(outputs, idx_to_char, blank_idx=0):
    """
    Decode CTC outputs to text using proper CTC collapse rules
    
    Args:
        outputs: (batch, seq_len, num_classes) or (batch, seq_len)
        idx_to_char: mapping from indices to characters
        blank_idx: index of blank token
    
    Returns:
        List of decoded strings
    """
    if len(outputs.shape) == 3:
        outputs = torch.argmax(outputs, dim=2)
    
    decoded_texts = []
    for output in outputs:
        chars = []
        prev_idx = None
        for idx in output.cpu().numpy():
            if idx == blank_idx:
                # Blank resets, allowing next char to be added even if same as before blank
                prev_idx = None
            elif idx != prev_idx:
                # Non-blank, different from previous non-blank
                if idx in idx_to_char:
                    chars.append(idx_to_char[idx])
                prev_idx = idx
            # else: same non-blank as previous, collapse it
        decoded_texts.append(''.join(chars))
    
    return decoded_texts


def calculate_levenshtein(predictions, targets):
    """
    Calculate average Levenshtein distance
    Ignoring case and spaces as per competition rules
    """
    total_distance = 0
    for pred, target in zip(predictions, targets):
        # Remove spaces and convert to lowercase
        pred_clean = pred.replace(' ', '').lower()
        target_clean = target.replace(' ', '').lower()
        
        distance = editdistance.eval(pred_clean, target_clean)
        total_distance += distance
    
    return total_distance / len(predictions) if predictions else 0


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))

def debug_decode(outputs, idx_to_char, blank_idx=0, show_top_k=3):
    """
    Debug CTC outputs - shows raw predictions including blanks
    
    Args:
        outputs: (batch, seq_len, num_classes) - logits/probs
        idx_to_char: mapping from indices to characters
        blank_idx: index of blank token
        show_top_k: show top k predictions per timestep
    """
    import torch.nn.functional as F
    
    probs = F.softmax(outputs, dim=2)
    argmax = torch.argmax(outputs, dim=2)
    
    for b in range(outputs.size(0)):
        print(f"\n=== Sample {b} ===")
        seq = argmax[b].cpu().numpy()
        prob_seq = probs[b].cpu().numpy()
        
        # Show collapsed sequence (what model outputs before CTC decode)
        raw_chars = []
        for idx in seq:
            if idx == blank_idx:
                raw_chars.append('_')  # underscore for blank
            elif idx in idx_to_char:
                raw_chars.append(idx_to_char[idx])
            else:
                raw_chars.append('?')
        
        print(f"Raw output (blanks as _): {''.join(raw_chars)}")
        
        # Show unique consecutive chars
        collapsed = []
        prev = None
        for c in raw_chars:
            if c != prev:
                collapsed.append(c)
            prev = c
        print(f"Collapsed (before blank removal): {''.join(collapsed)}")
        
        # Count blanks
        blank_count = sum(1 for idx in seq if idx == blank_idx)
        print(f"Blank count: {blank_count}/{len(seq)} ({100*blank_count/len(seq):.1f}%)")
        
        # Show positions where 'I' (or repeated char) appears
        # Find the index for capital I if it exists
        i_indices = [k for k, v in idx_to_char.items() if v in ['I', '𐐆']]
        if i_indices:
            print("Positions of 'I' variants: ", end="")
            for t, idx in enumerate(seq):
                if idx in i_indices:
                    print(f"t={t}(p={prob_seq[t, idx]:.2f}) ", end="")
            print()