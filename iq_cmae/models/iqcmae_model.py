import torch
import torch.nn as nn
import numpy as np
import copy
from functools import partial
from typing import Optional, List, Dict, Any, Tuple
from .mae_backbone import MaskedAutoencoderViT
from .pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed
from .modules import TransformerBlock, FeatureDecoder, NonLinearNeck


class _TeacherEncoder(nn.Module):
    """Minimal unmasked encoder view used by the EMA teacher."""

    def __init__(self, model):
        super().__init__()
        self.cls_token = model.cls_token
        self.patch_embed_const = model.patch_embed_const
        self.patch_embed_gaf = model.patch_embed_gaf
        self.patch_embed_spec = model.patch_embed_spec
        self.modality_blocks = model.modality_blocks
        self.fusion_proj = model.fusion_proj
        self.blocks = model.blocks
        self.norm = model.norm
        self.projector = model.projector

    def forward(self, x, channel_mask, modality, modality_pos, pos_embed):
        x = x * channel_mask
        branches = (
            self.patch_embed_const(x[:, :3]) + modality_pos[0],
            self.patch_embed_gaf(x[:, 3:5]) + modality_pos[1],
            self.patch_embed_spec(x[:, 5:6]) + modality_pos[2],
        )
        active = {"constellation": 0, "gaf": 1, "spectrogram": 2}.get(modality)
        if active is not None:
            x = branches[active]
            for block in self.modality_blocks[active]:
                x = block(x)
        else:
            encoded = []
            for branch, blocks in zip(branches, self.modality_blocks):
                for block in blocks:
                    branch = block(branch)
                encoded.append(branch)
            x = self.fusion_proj(torch.cat(encoded, dim=2))
        cls = (self.cls_token + pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class IQCMAE(MaskedAutoencoderViT):
    """
    IQCMAE: Multi-modal Contrastive Masked Autoencoder for IQ Data.
    Implements "Proper Fusion" (Mid-Fusion) and "Contrastive Gradient Stopping" (Last-K).
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=6,
                 embed_dim=192, depth=12, num_heads=3,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False,
                 # Contrastive learning parameters
                 contrastive_weight=2.5, temperature=0.07, projection_dim=256,
                 projector_hidden_dim=512, projector_layers=2,
                 predictor_hidden_dim=256, predictor_layers=2,
                 base_momentum=0.996,
                 # Contrastive gradient stopping (k parameter)
                 contrastive_last_k=4,
                 # Modality and shared layers support (S parameter)
                 modality_mask: Optional[str] = None,
                 shared_layers: int = 9,
                 # Contrastive pooling
                 contrastive_use_mask: bool = True,
                 head_type: str = 'ln',
                 fusion_type: str = 'concat'):
        
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss
        )

        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.contrastive_last_k = contrastive_last_k
        self.shared_layers_count = shared_layers
        self.contrastive_use_mask = contrastive_use_mask
        self.fusion_type = fusion_type

        modality = (modality_mask or "all").lower()
        if "+" in modality or "," in modality:
            modality = "all"
        channel_masks = {
            "all": [1, 1, 1, 1, 1, 1],
            "constellation": [1, 1, 1, 0, 0, 0],
            "gaf": [0, 0, 0, 1, 1, 0],
            "spectrogram": [0, 0, 0, 0, 0, 1],
        }
        if modality not in channel_masks:
            raise ValueError(f"unknown modality mask: {modality_mask}")
        self.modality_mask = modality
        self.register_buffer(
            "input_channel_mask",
            torch.tensor(channel_masks[modality], dtype=torch.float32).view(1, 6, 1, 1),
            persistent=False,
        )

        if not 0 <= shared_layers <= depth:
            raise ValueError("shared_layers must be between 0 and depth")
        if (depth - shared_layers) % 3:
            raise ValueError("depth - shared_layers must be divisible by 3 modalities")
        if not 0 <= contrastive_last_k <= shared_layers:
            raise ValueError("contrastive_last_k must be between 0 and shared_layers")

        # --------------------------------------------------------------------------
        # Proper Fusion Architecture Setup
        # --------------------------------------------------------------------------
        # 1. Separate Embeddings
        # Constellation (3 channels)
        self.patch_embed_const = PatchEmbed(img_size, patch_size, 3, embed_dim)
        # GAF (2 channels)
        self.patch_embed_gaf = PatchEmbed(img_size, patch_size, 2, embed_dim)
        # Spectrogram (1 channel)
        self.patch_embed_spec = PatchEmbed(img_size, patch_size, 1, embed_dim)
        
        # Separate Positional Embeddings
        num_patches = self.patch_embed.num_patches
        self.pos_embed_const = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.pos_embed_gaf = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.pos_embed_spec = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        # 2. Modality-Specific Blocks
        self.modality_specific_depth = (depth - shared_layers) // 3
        self.shared_depth = shared_layers
        
        self.modality_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(self.modality_specific_depth)
            ]) for _ in range(3) # 3 modalities
        ])
        
        # 3. Fusion Layer
        if fusion_type == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        # 4. Shared Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(self.shared_depth)
        ])

        # Projector & Predictor
        self.projector = NonLinearNeck(
            embed_dim, projector_hidden_dim, projection_dim, layers=projector_layers
        )
        self.predictor = NonLinearNeck(
            projection_dim, predictor_hidden_dim, projection_dim, layers=predictor_layers
        )

        self.initialize_proper_weights()
        self._init_target_network()
        self.criterion = nn.CrossEntropyLoss()

    def initialize_proper_weights(self):
        # Initialize pos embeds
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed_const.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_gaf.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_spec.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_target_network(self):
        """Initialize target network as a copy of online network."""
        self.target_encoder = copy.deepcopy(_TeacherEncoder(self)).requires_grad_(False)

    @torch.no_grad()
    def _momentum_update_target(self):
        online = list(_TeacherEncoder(self).parameters())
        target = list(self.target_encoder.parameters())
        torch._foreach_mul_(target, self.momentum)
        torch._foreach_add_(target, online, alpha=1.0 - self.momentum)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept checkpoints written before the teacher was grouped."""
        aliases = {
            "target_cls_token": "target_encoder.cls_token",
            "target_patch_embed_const": "target_encoder.patch_embed_const",
            "target_patch_embed_gaf": "target_encoder.patch_embed_gaf",
            "target_patch_embed_spec": "target_encoder.patch_embed_spec",
            "target_modality_blocks": "target_encoder.modality_blocks",
            "target_fusion_proj": "target_encoder.fusion_proj",
            "target_blocks": "target_encoder.blocks",
            "target_norm": "target_encoder.norm",
            "target_projector": "target_encoder.projector",
        }
        for key in list(state_dict):
            for old, new in aliases.items():
                old = prefix + old
                if key == old or key.startswith(old + "."):
                    state_dict[prefix + new + key[len(old):]] = state_dict.pop(key)
                    break
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward_contrastive(self, x1, x2):
        """InfoNCE between online predictions and stop-gradient EMA targets."""
        z1 = self.predictor(self.projector(x1))
        with torch.no_grad():
            z2 = self.target_encoder.projector(x2)
        
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        return nn.functional.cross_entropy(logits, labels)

    def forward_encoder(self, x, mask_ratio, gradient_stopping=False):
        x = x * self.input_channel_mask
        # 1. Split Input
        # x: [B, 6, H, W] -> Constellation (3), GAF (2), Spectrogram (1)
        x_const = x[:, :3, :, :]
        x_gaf = x[:, 3:5, :, :]
        x_spec = x[:, 5:6, :, :]

        # 2. Embed & Add Pos Embed (No CLS yet)
        x_c = self.patch_embed_const(x_const) + self.pos_embed_const
        x_g = self.patch_embed_gaf(x_gaf) + self.pos_embed_gaf
        x_s = self.patch_embed_spec(x_spec) + self.pos_embed_spec

        # Apply one spatial mask before the private stems so every modality
        # receives the same visible token set described in the manuscript.
        x_c, mask, ids_restore = self.random_masking(x_c, mask_ratio)
        len_keep = x_c.shape[1]
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :len_keep]
        gather = ids_keep.unsqueeze(-1).expand(-1, -1, x_g.shape[-1])
        x_g = torch.gather(x_g, dim=1, index=gather)
        x_s = torch.gather(x_s, dim=1, index=gather)

        active_index = {"constellation": 0, "gaf": 1, "spectrogram": 2}.get(
            self.modality_mask
        )
        if active_index is not None:
            x = (x_c, x_g, x_s)[active_index]
            for block in self.modality_blocks[active_index]:
                x = block(x)
        else:
            encoded = []
            for branch, blocks in zip((x_c, x_g, x_s), self.modality_blocks):
                for block in blocks:
                    branch = block(branch)
                encoded.append(branch)
            x = self.fusion_proj(torch.cat(encoded, dim=2))

        # 6. Append CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 7. Shared Blocks (with Gradient Stopping)
        if gradient_stopping:
            split_idx = len(self.blocks) - self.contrastive_last_k
            
            # Bottom layers (0 to L-K)
            for i in range(split_idx):
                x = self.blocks[i](x)
            
            # Split: one path for recon (attached), one for contrastive (detached)
            x_recon = x
            x_contrastive = x.detach()
            
            # Top layers (L-K to L)
            for i in range(split_idx, len(self.blocks)):
                blk = self.blocks[i]
                x_recon = blk(x_recon)
                x_contrastive = blk(x_contrastive)
                
            x_recon = self.norm(x_recon)
            x_contrastive = self.norm(x_contrastive)
            
            return x_recon, mask, ids_restore, x_contrastive
        else:
            # Standard forward (no split)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x, mask, ids_restore

    def forward_modality_loss(self, imgs, pred, mask):
        """Channel-normalized reconstruction loss summed across active modalities."""
        target = self.patchify(imgs)
        patch_pixels = target.shape[-1] // 6
        target = target.view(*target.shape[:2], patch_pixels, 6)
        pred = pred.view(*pred.shape[:2], patch_pixels, 6)
        if self.norm_pix_loss:
            mean = target.mean(dim=(-2, -1), keepdim=True)
            var = target.var(dim=(-2, -1), keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        active = self.input_channel_mask.view(-1).bool()
        losses = []
        for start, end in ((0, 3), (3, 5), (5, 6)):
            if active[start:end].any():
                error = (pred[..., start:end] - target[..., start:end]).pow(2)
                patch_loss = error.mean(dim=(-2, -1))
                losses.append((patch_loss * mask).sum() / mask.sum())
        return torch.stack(losses).sum()

    def forward(self, imgs, noisy_imgs=None, mask_ratio=0.75):
        imgs = imgs * self.input_channel_mask
        # Main pass (reconstruction)
        # We need both recon features (all gradients) and contrastive features (top-K gradients)
        latent_recon, mask, ids_restore, latent_contrastive = self.forward_encoder(imgs, mask_ratio, gradient_stopping=True)
        
        pred = self.forward_decoder(latent_recon, ids_restore)
        loss_recon = self.forward_modality_loss(imgs, pred, mask)
        
        loss_contrastive = torch.tensor(0.0, device=imgs.device)
        if noisy_imgs is not None:
             if self.training:
                 self._momentum_update_target()
             latent_noisy_contrastive = self.target_encoder(
                 noisy_imgs, self.input_channel_mask, self.modality_mask,
                 (self.pos_embed_const, self.pos_embed_gaf, self.pos_embed_spec),
                 self.pos_embed,
             )
             
             z1 = latent_contrastive[:, 0]
             z2 = latent_noisy_contrastive[:, 0]
             
             loss_contrastive = self.forward_contrastive(z1, z2) * self.contrastive_weight

        return loss_recon + loss_contrastive, loss_recon, loss_contrastive, pred, mask

    def update_momentum(self, epoch, max_epochs):
        self.momentum = 1. - (1. - self.base_momentum) * (np.cos(np.pi * epoch / max_epochs) + 1) * 0.5


CorrectedProperCMAE = IQCMAE
