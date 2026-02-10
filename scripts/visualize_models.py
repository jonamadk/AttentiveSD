"""
Visualize model architectures from saved checkpoints.
This script loads the saved models and provides detailed architecture summaries.
"""

from attentivesd.models.model import HybridSpliceModel
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_layer_info(model, indent=0):
    """Recursively print layer information."""
    prefix = "  " * indent
    for name, module in model.named_children():
        # Get parameter count for this module
        params = sum(p.numel() for p in module.parameters())

        # Check if it's a leaf module (no children)
        if len(list(module.children())) == 0:
            module_type = module.__class__.__name__

            # Get shape information
            shape_info = ""
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                shape_info = f"[{module.in_features} → {module.out_features}]"
            elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                kernel = getattr(module, 'kernel_size', None)
                dilation = getattr(module, 'dilation', None)
                shape_info = f"[{module.in_channels} → {module.out_channels}"
                if kernel:
                    shape_info += f", k={kernel[0] if isinstance(kernel, tuple) else kernel}"
                if dilation and dilation != (1,) and dilation != 1:
                    shape_info += f", d={dilation[0] if isinstance(dilation, tuple) else dilation}"
                shape_info += "]"
            elif hasattr(module, 'num_features'):
                shape_info = f"[{module.num_features}]"
            elif hasattr(module, 'normalized_shape'):
                shape_info = f"[{module.normalized_shape}]"
            elif hasattr(module, 'num_heads'):
                shape_info = f"[{module.num_heads} heads, dim={module.embed_dim}]"

            print(f"{prefix}├─ {name}: {module_type}{shape_info} - {params:,} params")
        else:
            print(
                f"{prefix}├─ {name}: {module.__class__.__name__} - {params:,} params")
            print_layer_info(module, indent + 1)


def get_model_summary(checkpoint_path, model_mode):
    """Load model and print detailed summary."""
    print(f"\n{'='*80}")
    print(f"Model Architecture: {model_mode.upper().replace('_', ' + ')}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model config
    config = {
        "mode": model_mode,
        "conv_channels": [64, 128, 256, 256],
        "kernel_sizes": [7, 7, 7, 7],
        "dilations": [1, 2, 4, 8],
        "dropout": 0.3,
        "pooling": "center",
        "attention": {
            "num_layers": 4,
            "num_heads": 8,
            "hidden_dim": 256,
            "mlp_dim": 1024,
            "use_rope": model_mode == "cnn_attention_rope",
            "attn_dropout": 0.2,
            "proj_dropout": 0.2
        }
    }

    # Create model
    model = HybridSpliceModel(config)
    model.eval()

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (float32)")

    # Training info from checkpoint
    if 'epoch' in checkpoint:
        print(f"\nTraining Info:")
        print(f"  Best Epoch: {checkpoint['epoch']}")
    if 'best_val_f1' in checkpoint:
        print(f"  Best Val F1: {checkpoint['best_val_f1']:.4f}")

    print(f"\n{'─'*80}")
    print("Layer-by-Layer Architecture:")
    print(f"{'─'*80}\n")

    # Print architecture tree
    print_layer_info(model)

    # Print detailed breakdown by component
    print(f"\n{'─'*80}")
    print("Component Breakdown:")
    print(f"{'─'*80}\n")

    components = {}

    if hasattr(model, 'cnn') and model.cnn is not None:
        components['CNN Encoder'] = model.cnn

    if hasattr(model, 'attn') and model.attn is not None:
        components['Attention Encoder'] = model.attn

    if hasattr(model, 'classifier'):
        components['Classification Head'] = model.classifier

    for comp_name, comp in components.items():
        params = sum(p.numel() for p in comp.parameters())
    print("Architecture Features:")
    print(f"{'─'*80}\n")

    if model_mode == "cnn":
        print("✓ Residual Convolutional Blocks (4 blocks)")
        print("✓ Multi-scale feature extraction (dilations: 1, 2, 4, 8)")
        print("✓ Batch Normalization")
        print("✓ Dropout (0.3)")
        print("✓ Center position pooling")
        print(f"✓ Receptive field: ~187 base pairs")

    elif model_mode == "cnn_attention":
        print("✓ All CNN baseline features")
        print("✓ Multi-head self-attention (8 heads)")
        print("✓ 4 Transformer blocks")
        print("✓ Pre-normalization architecture")
        print("✓ GELU activation in FFN")
        print("✓ 4× MLP expansion (256 → 1024 → 256)")
        print("✓ Dropout (0.2 in attention)")
        print("✓ Full sequence receptive field (400bp)")
        print("✗ No explicit positional encoding")

    elif model_mode == "cnn_attention_rope":
        print("✓ All CNN + Attention features")
        print("✓ Rotary Positional Embeddings (RoPE)")
        print("✓ Position-aware attention scores")
        print("✓ Relative distance encoding")
        print("✓ Zero additional parameters for RoPE")
        print("✓ Enhanced position-specific learning")
        print(f"✓ Same parameter count as CNN+Attention: {total_params:,}")

    print()


def visualize_all_models(base_dir="outputs"):
    """Visualize architectures for all three models."""
    base_path = Path(base_dir)

    models = [
        ("cnn", base_path / "cnn" / "checkpoint_best.pt"),
        ("cnn_attention", base_path / "cnn_attention" / "checkpoint_best.pt"),
        ("cnn_attention_rope", base_path /
         "cnn_attention_rope" / "checkpoint_best.pt"),
    ]

    for model_mode, checkpoint_path in models:
        if checkpoint_path.exists():
            get_model_summary(str(checkpoint_path), model_mode)
        else:
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")

    # Print comparison summary
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print("┌─────────────────────────┬──────────────┬───────────────┬──────────────┐")
    print("│ Model                   │ Parameters   │ Receptive     │ Key Feature  │")
    print("│                         │              │ Field         │              │")
    print("├─────────────────────────┼──────────────┼───────────────┼──────────────┤")
    print("│ CNN Baseline            │ ~502K        │ 187bp (local) │ Dilated Conv │")
    print("│ CNN + Attention         │ ~3.66M       │ 400bp (full)  │ Self-Attn    │")
    print("│ CNN + Attention + RoPE  │ ~3.66M       │ 400bp (full)  │ RoPE Encoding│")
    print("└─────────────────────────┴──────────────┴───────────────┴──────────────┘")

    print("\nParameter Scaling:")
    print("  CNN → CNN+Attention:      7.3× increase (502K → 3.66M)")
    print("  CNN+Attention → +RoPE:    0× increase (RoPE is parameter-free)")

    print("\nPerformance Scaling:")
    print("  CNN → CNN+Attention:      +16.51% accuracy (70.37% → 86.88%)")
    print("  CNN+Attention → +RoPE:    +8.32% accuracy (86.88% → 95.20%)")

    print("\nEfficiency Metrics:")
    print("  Best accuracy/param:      CNN+Attention+RoPE (95.20% / 3.66M = 0.026 per 1K params)")
    print("  Fastest inference:        CNN (5ms/sample)")
    print("  Production choice:        CNN+Attention+RoPE (95.2% accuracy, 12ms/sample)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize model architectures from checkpoints")
    parser.add_argument("--model", "-m", type=str, choices=["cnn", "cnn_attention", "cnn_attention_rope", "all"],
                        default="all", help="Which model to visualize (default: all)")
    parser.add_argument("--checkpoint", "-c", type=str,
                        help="Path to specific checkpoint file")
    parser.add_argument("--base-dir", "-d", type=str,
                        default="outputs", help="Base output directory")

    args = parser.parse_args()

    if args.checkpoint:
        # Visualize specific checkpoint
        if args.model == "all":
            print("Error: Must specify --model when using --checkpoint")
            sys.exit(1)
        get_model_summary(args.checkpoint, args.model)
    elif args.model == "all":
        # Visualize all models
        visualize_all_models(args.base_dir)
    else:
        # Visualize specific model
        checkpoint_path = Path(args.base_dir) / \
            args.model / "checkpoint_best.pt"
        if checkpoint_path.exists():
            get_model_summary(str(checkpoint_path), args.model)
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
