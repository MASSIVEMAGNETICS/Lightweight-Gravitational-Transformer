"""
Lightweight Gravitational Transformer (LGT)
============================================
A physics-aware transformer architecture using gravitational attention,
designed for edge deployment and VictorOS cognitive-runtime integration.

Quick start
-----------
>>> import torch
>>> from lightweight_gravitational_transformer import LightweightGravitationalTransformer
>>> model = LightweightGravitationalTransformer(vocab_size=1000, dim_model=64)
>>> x = torch.randint(0, 1000, (1, 16))
>>> output, _ = model(x)
>>> output.shape
torch.Size([1, 16, 64])
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"
__license__ = "MIT"

# Core attention
from gravitational_attention import (
    GravitationalAttentionHead,
    MultiHeadGravitationalAttention,
)

# Position encodings
from fractal_position_embedding import FractalPositionEmbedding
from lightweight_gravitational_transformer import (
    CurvedPositionEmbedding,
    LightweightGravitationalBlock,
    LightweightGravitationalTransformer,
)

# VictorOS integration
from victorcos_module import (
    Ledger,
    LedgerEntry,
    MirrorLayer,
    VictorOSBaseModule,
    VictorOSModuleMetadata,
    LGTVictorOSModule,
    MorphicVictorAgent,
    victoros_module,
)

# Morphic Cognitive Engine
from octonion_pos_embedding import (
    OctonionEmbedding,
    octonion_distance,
    GravitationalOctonionPosition,
)
from polymorphic_attention_orchestrator import (
    PHASE_CONFIG,
    PolymorphicAttentionOrchestrator,
)
from training_containment import (
    MorphicContainmentConfig,
    MorphicContainmentProtocol,
)

# Training
from training import (
    ContainmentConfig,
    ContainmentProtocol,
    MetaCurvatureScheduler,
    TrainingConfig,
    TrainingLoop,
)

# Tri-model
from tri_model import (
    CrossGravitationalFusion,
    TriModelTransformer,
)

# Edge export
from export_edge_model import (
    PRESETS,
    build_model,
    export_edge_model,
    export_torchscript,
    quantize_dynamic,
    save_checkpoint,
)

__all__ = [
    # Version
    "__version__",
    # Attention
    "GravitationalAttentionHead",
    "MultiHeadGravitationalAttention",
    # Position encodings
    "FractalPositionEmbedding",
    "CurvedPositionEmbedding",
    # Transformer blocks
    "LightweightGravitationalBlock",
    "LightweightGravitationalTransformer",
    # VictorOS
    "Ledger",
    "LedgerEntry",
    "MirrorLayer",
    "VictorOSBaseModule",
    "VictorOSModuleMetadata",
    "LGTVictorOSModule",
    "MorphicVictorAgent",
    "victoros_module",
    # Morphic Cognitive Engine
    "OctonionEmbedding",
    "octonion_distance",
    "GravitationalOctonionPosition",
    "PHASE_CONFIG",
    "PolymorphicAttentionOrchestrator",
    "MorphicContainmentConfig",
    "MorphicContainmentProtocol",
    # Training
    "ContainmentConfig",
    "ContainmentProtocol",
    "MetaCurvatureScheduler",
    "TrainingConfig",
    "TrainingLoop",
    # Tri-model
    "CrossGravitationalFusion",
    "TriModelTransformer",
    # Export
    "PRESETS",
    "build_model",
    "export_edge_model",
    "export_torchscript",
    "quantize_dynamic",
    "save_checkpoint",
]
