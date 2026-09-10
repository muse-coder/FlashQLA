"""PPU implementation hooks for the shared FlashQLA chunk entry point."""

from .production_fastpath import try_production_fastpath
from .torch_backend import (
    CHUNK_SIZE,
    official_chunk_backward,
    official_chunk_forward,
    official_chunk_gated_delta_rule,
)
__all__ = [
    "CHUNK_SIZE",
    "official_chunk_backward",
    "official_chunk_forward",
    "official_chunk_gated_delta_rule",
    "try_production_fastpath",
]
