"""PPU primitives for the shared FlashQLA chunk entry point."""

from .ops import (
    CHUNK_SIZE,
    chunk_local_cumsum,
    correct_initial_states,
    correct_terminal_states,
    fused_gdr_bwd,
    fused_gdr_dh,
    fused_gdr_fwd,
    fused_gdr_h,
    get_warmup_chunks,
    get_warmup_chunks_bidi,
    group_reduce_vector,
    kkt_solve,
)

__all__ = [
    "CHUNK_SIZE",
    "chunk_local_cumsum",
    "group_reduce_vector",
    "kkt_solve",
    "fused_gdr_fwd",
    "fused_gdr_h",
    "fused_gdr_bwd",
    "fused_gdr_dh",
    "get_warmup_chunks",
    "get_warmup_chunks_bidi",
    "correct_initial_states",
    "correct_terminal_states",
]
