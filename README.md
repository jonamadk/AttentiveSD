# AttentiveSD

AttentiveSD is a CNN-Attention hybrid for splice site prediction. It keeps the local motif sensitivity of 1D CNNs and adds a lightweight self-attention stage with Rotary Positional Embeddings (RoPE) to model long-range dependencies.

## Features
- Shallow residual CNN blocks with optional dilations for motif-level signals
- Multi-head self-attention for distal context modeling
- RoPE or learnable positional handling via attention configuration
- CNN-only, CNN-Attention, and Attention-only ablations

## Quickstart (local data)

### Preprocess
```bash
PYTHONPATH=src python scripts/preprocess_data.py --dataset balanced --organism hs --site donor --split train
PYTHONPATH=src python scripts/preprocess_data.py --dataset balanced --organism hs --site donor --split test
```

### Train
```bash
PYTHONPATH=src python scripts/train.py --dataset balanced --organism hs --site donor --mode cnn_attention
```

### Evaluate
```bash
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/checkpoint_best.pt --dataset balanced --organism hs --site donor --mode cnn_attention
```

### One-shot pipeline
```bash
PYTHONPATH=src python scripts/main.py --dataset balanced --organism hs --site donor --mode cnn_attention
```

## Quickstart
1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Preprocess to npz for faster loading (uses the local data/ folder):

```bash
PYTHONPATH=src python scripts/preprocess_data.py --dataset balanced --organism hs --site donor --split train
PYTHONPATH=src python scripts/preprocess_data.py --dataset balanced --organism hs --site donor --split test
```

3) Train:

```bash
PYTHONPATH=src python scripts/train.py --dataset balanced --organism hs --site donor --mode cnn_attention
```

To resume after a crash, pass a checkpoint (typically outputs/checkpoint_last.pt):

```bash
PYTHONPATH=src python scripts/train.py --dataset balanced --organism hs --site donor --mode cnn_attention --resume outputs/checkpoint_last.pt
```

4) Evaluate (requires a checkpoint path):

```bash
PYTHONPATH=src python scripts/evaluate.py --checkpoint path/to/model.pt --dataset balanced --organism hs --site donor --mode cnn_attention
```

Training saves checkpoints by default to outputs/checkpoint_best.pt and outputs/checkpoint_last.pt.

## Configuration
Defaults live in configs/default.yaml. You can override key settings via CLI flags like --epochs, --batch-size, and --lr.

Model modes:
- cnn: CNN-only baseline
- attention: Attention-only baseline
- cnn_attention: CNN + attention hybrid

## Data Notes
The loader expects one-hot encoded sequences per line and matching label files in the local data/ folder following the balanced/imbalanced organism layout. If a label line contains a per-position vector, the center position is used as the binary label.

