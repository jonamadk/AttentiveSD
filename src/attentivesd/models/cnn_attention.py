from dataclasses import dataclass
from typing import List

import torch
from torch import nn


def compute_padding(kernel_size: int, dilation: int) -> int:
    return ((kernel_size - 1) // 2) * dilation


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = compute_padding(kernel_size, dilation)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x + residual)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, channels: List[int], kernel_sizes: List[int], dilations: List[int], dropout: float):
        super().__init__()
        blocks = []
        in_ch = 4
        for out_ch, kernel, dilation in zip(channels, kernel_sizes, dilations):
            blocks.append(ResidualConvBlock(in_ch, out_ch, kernel, dilation, dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.stack((freqs.sin(), freqs.cos()), dim=-1)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    sin = rope_cache[..., 0]
    cos = rope_cache[..., 1]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    sin = sin.repeat_interleave(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1)
    return (x * cos) + (x_rot * sin)


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float, proj_dropout: float, use_rope: bool):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            rope_cache = build_rope_cache(seq_len, self.head_dim, x.device)
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.proj(out)
        return self.proj_dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, attn_dropout: float, proj_dropout: float, use_rope: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttentionRoPE(embed_dim, num_heads, attn_dropout, proj_dropout, use_rope)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_layers: int, num_heads: int, mlp_dim: int, attn_dropout: float, proj_dropout: float, use_rope: bool
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, attn_dropout, proj_dropout, use_rope)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class AttentionConfig:
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    use_rope: bool
    attn_dropout: float
    proj_dropout: float


@dataclass
class ModelConfig:
    mode: str
    conv_channels: List[int]
    kernel_sizes: List[int]
    dilations: List[int]
    dropout: float
    attention: AttentionConfig
    pooling: str


class HybridSpliceModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        attention = AttentionConfig(**config["attention"])
        model_config = ModelConfig(attention=attention, **{k: v for k, v in config.items() if k != "attention"})
        self.config = model_config

        self.mode = model_config.mode
        self.pooling = model_config.pooling

        if self.mode in {"cnn", "cnn_attention"}:
            self.cnn = CNNEncoder(
                channels=model_config.conv_channels,
                kernel_sizes=model_config.kernel_sizes,
                dilations=model_config.dilations,
                dropout=model_config.dropout,
            )
            embed_dim = self.cnn.out_channels
        else:
            self.cnn = None
            embed_dim = attention.hidden_dim
            self.input_proj = nn.Linear(4, embed_dim)

        if self.mode in {"attention", "cnn_attention"}:
            self.attn = AttentionEncoder(
                embed_dim=embed_dim,
                num_layers=attention.num_layers,
                num_heads=attention.num_heads,
                mlp_dim=attention.mlp_dim,
                attn_dropout=attention.attn_dropout,
                proj_dropout=attention.proj_dropout,
                use_rope=attention.use_rope,
            )
        else:
            self.attn = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return x.mean(dim=1)
        center = x.shape[1] // 2
        return x[:, center, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"cnn", "cnn_attention"}:
            x = x.transpose(1, 2)
            x = self.cnn(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_proj(x)

        if self.attn is not None:
            x = self.attn(x)

        pooled = self.pool(x)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
