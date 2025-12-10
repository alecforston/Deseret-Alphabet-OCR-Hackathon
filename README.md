# Deseret Alphabet OCR with CRNN

CRNN (Convolutional Recurrent Neural Network) implementation for transcribing Deseret Alphabet manuscript images to Unicode text.

## Project Structure

```
deseret-ocr/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── labels/          # Training labels (.txt files)
│   ├── test/
│   │   └── images/          # Test images (unlabeled)
├── src/
│   ├── dataset.py          # PyTorch Dataset classes
│   ├── model.py            # CRNN architecture
│   ├── train.py            # Training script
│   ├── inference.py        # Inference and submission generation
│   └── utils.py            # Helper functions
├── configs/
│   └── config.yaml         # Configuration file
├── models/                  # Saved model checkpoints
├── submissions/            # Generated submission files
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your data according to the structure above.

## Workflow

### 1. Configure Training

Edit `configs/config.yaml` to set:
- Data paths
- Preprocessing dimensions (based on your analysis)
- Training hyperparameters

### 2. Train Model

```bash
python src/train.py --config configs/config.yaml
```

To resume training from a checkpoint:
```bash
python src/train.py --config configs/config.yaml --resume models/crnn_best.pth
```

### 3. Generate Output

```bash
python src/inference.py \
    --config configs/config.yaml \
    --checkpoint models/crnn_best.pth \
    --output submissions/submission.csv
```

## Model Architecture

**CRNN Components:**

1. **CNN Backbone**: Extracts visual features from images
   - 5 convolutional blocks with batch normalization
   - MaxPooling to reduce spatial dimensions
   - Output: Feature sequence representing horizontal positions

2. **Bidirectional LSTM**: Captures sequential context
   - Reads features left-to-right and right-to-left
   - Learns character dependencies and context

3. **CTC Loss**: Handles alignment between images and text
   - No manual character segmentation required
   - Automatically learns alignment during training

## About CRNN and CTC (Connectionist Temporal Classification)

**CRNN:**
- Handles variable-width line images naturally
- No need to segment individual characters
- Captures both local (CNN) and sequential (RNN) patterns

**CTC:**
- Solves the alignment problem: image width ≠ text length
- Allows model to output blanks and repeated characters
- Example: "hh_eee_ll_oo" → "hello" (where _ = blank token)

**Whitespace handling:**
CRNN with CTC can handle whitespace in images because:
- The RNN learns to output blank tokens for whitespace regions
- Training labels include space characters in the vocabulary
- The model learns the visual pattern of word spacing