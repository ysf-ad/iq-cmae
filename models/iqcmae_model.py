import torch
import torch.nn as nn
import numpy as np
from functools import partial
from typing import Optional, List, Dict, Any, Tuple
from .mae_backbone import MaskedAutoencoderViT
from .pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed
from .modules import TransformerBlock, FeatureDecoder, NonLinearNeck

class IQCMAE(MaskedAutoencoderViT):
    """
    IQCMAE: Multi-modal Contrastive Masked Autoencoder for IQ Data.
    Implements Mid-Fusion and Contrastive Gradient Stopping (Last-K).
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

        # --------------------------------------------------------------------------
        # Architecture Setup
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
        self.modality_specific_depth = depth - shared_layers
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

        self._init_target_network()
        self.criterion = nn.CrossEntropyLoss()
        self.initialize_proper_weights()

    def initialize_proper_weights(self):
        # Initialize pos embeds
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed_const.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_gaf.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_spec.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_target_network(self):
        """Initialize target network as a copy of online network."""
        pass

    def forward_contrastive(self, x1, x2):
        """InfoNCE contrastive loss calculation."""
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        batch_size = z1.shape[0]
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z1.device),
            torch.arange(0, batch_size, device=z1.device)
        ])
        
        positives = similarity_matrix[torch.arange(2 * batch_size, device=z1.device), pos_indices]
        
        numerator = torch.exp(positives)
        denominator = torch.exp(similarity_matrix).sum(dim=1)
        
        loss = -torch.log(numerator / denominator).mean()
        
        return loss

    def forward_encoder(self, x, mask_ratio, gradient_stopping=False):
        # 1. Split Input
        # x: [B, 6, H, W] -> Constellation (3), GAF (2), Spectrogram (1)
        x_const = x[:, :3, :, :]
        x_gaf = x[:, 3:5, :, :]
        x_spec = x[:, 5:6, :, :]

        # 2. Embed & Add Pos Embed (No CLS yet)
        x_c = self.patch_embed_const(x_const) + self.pos_embed_const
        x_g = self.patch_embed_gaf(x_gaf) + self.pos_embed_gaf
        x_s = self.patch_embed_spec(x_spec) + self.pos_embed_spec

        # 3. Modality-Specific Blocks
        for blk in self.modality_blocks[0]: x_c = blk(x_c)
        for blk in self.modality_blocks[1]: x_g = blk(x_g)
        for blk in self.modality_blocks[2]: x_s = blk(x_s)

        # 4. Fusion
        # Concat along channel dimension (B, N, 3*D)
        x_fused = torch.cat([x_c, x_g, x_s], dim=2)
        x = self.fusion_proj(x_fused)

        # 5. Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

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

    def forward(self, imgs, noisy_imgs=None, mask_ratio=0.75):
        # Main pass (reconstruction)
        # We need both recon features (all gradients) and contrastive features (top-K gradients)
        latent_recon, mask, ids_restore, latent_contrastive = self.forward_encoder(imgs, mask_ratio, gradient_stopping=True)
        
        pred = self.forward_decoder(latent_recon, ids_restore)
        loss_recon = self.forward_loss(imgs, pred, mask)
        
        loss_contrastive = torch.tensor(0.0, device=imgs.device)
        if noisy_imgs is not None:
             # Contrastive pass
             # Only use contrastive features (top-K gradients)
             _, _, _, latent_noisy_contrastive = self.forward_encoder(noisy_imgs, mask_ratio=0.0, gradient_stopping=True)
             
             z1 = latent_contrastive[:, 0]
             z2 = latent_noisy_contrastive[:, 0]
             
             loss_contrastive = self.forward_contrastive(z1, z2) * self.contrastive_weight

        return loss_recon + loss_contrastive, loss_recon, loss_contrastive, pred, mask

    def update_momentum(self, epoch, max_epochs):
        self.momentum = 1. - (1. - self.base_momentum) * (np.cos(np.pi * epoch / max_epochs) + 1) * 0.5
