import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

class TransformerBlock(nn.Module):
    """Standard Transformer Block wrapper."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block = Block(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer
        )

    def forward(self, x):
        return self.block(x)

class FeatureDecoder(nn.Module):
    """Decoder for reconstructing features from latent representations."""
    def __init__(self, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=True, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

    def forward(self, x):
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

class NonLinearNeck(nn.Module):
    """Non-linear projection head for contrastive learning."""
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for i in range(layers - 1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, out_dim, bias=False))
        self.layers.append(nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
