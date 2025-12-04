"""
Model components for IQ-CMAE.

Expose the maintained model architectures and supporting utilities.
"""

from .iqcmae_model import IQCMAE
# Avoid importing MidFusionCMAE at module import time to prevent parsing errors
# from .mid_fusion_cmae import MidFusionCMAE
from .pos_embed import get_2d_sincos_pos_embed

__all__ = [
    "CorrectedProperCMAE",
    "get_2d_sincos_pos_embed",
]
