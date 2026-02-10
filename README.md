# AttentiveSD

AttentiveSD is a deep learning framework for splice site prediction with multiple architecture options: CNN-only, CNN+Attention, and CNN+Attention+RoPE. It combines local motif detection through residual CNNs with long-range dependency modeling via multi-head self-attention and Rotary Positional Embeddings (RoPE).

## Features

### Model Architectures
- **CNN**: CNN-only baseline with residual blocks and dilated convolutions
- **CNN+Attention**: Hybrid model combining CNN feature extraction with multi-head self-attention
- **CNN+Attention+RoPE**: Full model with Rotary Positional Embeddings for enhanced positional encoding

### Training Enhancements
- **Early Stopping**: Automatic termination when validation metrics plateau (default: 10 epochs patience)
- **Learning Rate Scheduling**: ReduceLROnPlateau or CosineAnnealing schedulers
- **Gradient Clipping**: Prevents exploding gradients (max norm: 1.0)
- **Comprehensive Logging**: Timestamped training logs with epoch-wise metrics

### Evaluation & Visualization
- **Confusion Matrix**: Heatmap visualization of model predictions
- **Classification Report**: Per-class precision, recall, and F1-score
- **Training Curves**: Loss, accuracy, and F1 progression (train/val)
- **ROC Curve**: Receiver Operating Characteristic with AUC score
- **Precision-Recall Curve**: PR curve with Average Precision
- **Performance Comparison**: Train/val/test metrics side-by-side

### Data Management
- **Flexible Sample Control**: Choose exact number of training samples for quick experiments
- **Automatic Test Sizing**: Test set automatically sized to 20% of training samples
- **Efficient Preprocessing**: Data cached as compressed .npz files

## Quick Start

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### How to Run the Models

Run the complete interactive pipeline with a single command:

```bash
PYTHONPATH=src .venv/bin/python scripts/main.py
```

This interactive pipeline will guide you through:
1. **Data Processing**: Automatically preprocesses train and test splits
2. **Dataset Info**: Shows available training and test samples
3. **Sample Selection**: Choose number of training samples (or use all)
4. **Model Selection**: Pick architecture (CNN, CNN+Attention, CNN+Attention+RoPE, or train all)
5. **Training & Evaluation**: Automatic training followed by comprehensive evaluation

**Example Session:**
```
Dataset Information:
  Training samples: 16,000
  Test samples: 9,000

How many training samples do you want to use?
  Enter a number (1-16,000) or press Enter to use all samples
Training samples [16,000]: 5000
✓ Will use 5,000 training samples
✓ Will use 1,000 test samples (20% of training)

Select model architecture:
  1) cnn                - CNN-only (no attention)
  2) cnn_attention      - CNN + Multi-head Attention (no RoPE)
  3) cnn_attention_rope - CNN + Multi-head Attention + RoPE
  4) all                - Train and evaluate all models sequentially

Enter choice (1/2/3/4): 3
```

## Output Structure

Each model saves to its own directory for easy comparison:

```
outputs/
├── cnn/
│   ├── checkpoint_best.pt              # Best model (by validation F1)
│   ├── checkpoint_last.pt              # Latest epoch checkpoint
│   ├── training_log.txt                # Timestamped training log
│   ├── training_history.json           # Epoch metrics (JSON)
│   ├── classification_report.txt       # Per-class metrics
│   ├── confusion_matrix.png            # Confusion matrix heatmap
│   ├── training_curves.png             # Train/val loss, acc, F1
│   ├── roc_curve.png                   # ROC curve with AUC
│   ├── precision_recall_curve.png      # PR curve with AP
│   └── train_val_test_comparison.png   # Performance comparison
├── cnn_attention/
│   └── [same files]
└── cnn_attention_rope/
    └── [same files]
```

## Configuration

Edit `configs/default.yaml` or use CLI flags to customize:

### Model Configuration
```yaml
model:
  mode: cnn_attention_rope  # cnn, cnn_attention, cnn_attention_rope
  conv_channels: [64, 128, 128]
  kernel_sizes: [9, 7, 5]
  dilations: [1, 2, 4]
  dropout: 0.3
  attention:
    num_layers: 2
    num_heads: 4
    hidden_dim: 128
    mlp_dim: 256
    attn_dropout: 0.2
    proj_dropout: 0.2
  pooling: center  # center or mean
```

### Training Configuration
```yaml
train:
  batch_size: 64
  epochs: 10
  lr: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  grad_clip: 1.0
  scheduler: plateau  # plateau, cosine, or none
```

### CLI Overrides
```bash
--mode cnn_attention_rope   # Model architecture
--epochs 50                 # Training epochs
--batch-size 128            # Batch size
--lr 0.0005                 # Learning rate
--max-samples 5000          # Limit training samples
--max-test-samples 1000     # Limit test samples
```

## Model Architectures

### 1. CNN (Baseline)
Pure convolutional architecture with residual blocks:
- Shallow residual CNN blocks with batch normalization
- Dilated convolutions for expanded receptive fields
- Dropout for regularization
- Center or mean pooling

### 2. CNN+Attention
Hybrid architecture combining CNNs with attention:
- CNN encoder for local motif extraction
- Multi-head self-attention for long-range dependencies
- LayerNorm and MLP blocks
- **No positional embeddings** (relies on CNN positional info)

### 3. CNN+Attention+RoPE (Full Model)
Enhanced hybrid with rotary positional embeddings:
- All features from CNN+Attention
- **Rotary Positional Embeddings (RoPE)** for explicit position encoding
- Better handling of sequence position information
- Improved performance on long-range dependencies

## Training Tips

### Quick Experiments
```bash
# Fast iteration with limited samples
echo -e "1000\n1" | PYTHONPATH=src .venv/bin/python scripts/main.py
# Uses 1000 training samples, 200 test samples (20%), CNN model
```

### Full Training
```bash
# Train all models with full dataset
echo -e "\n4" | PYTHONPATH=src .venv/bin/python scripts/main.py
# Press Enter for all samples, select option 4 for all models
```

### Hyperparameter Tuning
- Start with fewer samples (e.g., 1000-5000) for quick iterations
- Use early stopping to prevent overfitting
- Monitor validation metrics closely
- Increase dropout if overfitting occurs (train >> val performance)
- Try different learning rates (0.0001 to 0.001)

## Data Format

### Input Files
The pipeline expects:
- One-hot encoded sequences (4 channels: A, C, G, T)
- Binary labels (donor vs non-donor sites)
- Organized by dataset/organism/site:
  ```
  data/balanced/hs/
  ├── train_donor_hs
  ├── train_donor_hs_lbl
  ├── test_donor_hs
  └── test_donor_hs_lbl
  ```

### Supported Organisms
- `hs`: Homo sapiens (human)
- `at`: Arabidopsis thaliana
- `d_mel`: Drosophila melanogaster
- `c_elegans`: Caenorhabditis elegans
- `oriza`: Oryza sativa (rice)

### Supported Sites
- `donor`: Donor splice sites (GT sites)
- `acceptor`: Acceptor splice sites (AG sites)

## Advanced Usage

### Training with Custom Config
```python
# Create custom config
import yaml

config = {
    'model': {
        'mode': 'cnn_attention_rope',
        'conv_channels': [128, 256, 256],  # Larger model
        'dropout': 0.4
    },
    'train': {
        'epochs': 50,
        'lr': 0.0005,
        'early_stopping_patience': 15
    }
}

with open('configs/custom.yaml', 'w') as f:
    yaml.dump(config, f)

# Train with custom config
PYTHONPATH=src .venv/bin/python scripts/train.py --config configs/custom.yaml
```

### Batch Evaluation
```bash
# Evaluate all saved models
for model in outputs/*/checkpoint_best.pt; do
    echo "Evaluating $model"
    PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint $model
done
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config or use `--batch-size 32`
- Limit training samples with `--max-samples`
- Use smaller model (reduce `conv_channels`)

### Overfitting (train >> val performance)
- Increase dropout values (0.3 to 0.5)
- Reduce model capacity
- Use more training data
- Enable early stopping (already on by default)
- Increase weight decay

### Poor Performance
- Try different model architectures (compare all 3)
- Increase training epochs
- Tune learning rate
- Check data balance (use balanced dataset)
- Verify data preprocessing

## Citation

If you use AttentiveSD in your research, please cite:

```bibtex
@software{attentivesd2026,
  title={AttentiveSD: CNN-Attention-RoPE Hybrid for Splice Site Prediction},
  author={Manoj Adhikari},
  year={2026},
  url={https://github.com/jonamadk/AttentiveSD}
}
```

## License

MIT License - see LICENSE file for details

