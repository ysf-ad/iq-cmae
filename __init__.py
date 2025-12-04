"""
IQ-CMAE package namespace.

Provides optional convenience re-exports. Modules that are not present are skipped so
importing ``iq_cmae`` remains robust when legacy files are removed.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = []

try:
    from .models.mid_fusion_cmae import MidFusionCMAE  # noqa: F401
    __all__.append("MidFusionCMAE")
except Exception:  # pragma: no cover - optional dependency
    MidFusionCMAE = None

try:
    from .data.on_the_fly_multimodal_dataset import OnTheFlyMultimodalDataset  # noqa: F401
    __all__.append("OnTheFlyMultimodalDataset")
except Exception:  # pragma: no cover
    OnTheFlyMultimodalDataset = None

try:
    from .data.ne_data_full_dataset import NEDataFullDataset, build_ne_data_full_dataset  # noqa: F401
    __all__.extend(["NEDataFullDataset", "build_ne_data_full_dataset"])
except Exception:  # pragma: no cover
    NEDataFullDataset = None
    build_ne_data_full_dataset = None
