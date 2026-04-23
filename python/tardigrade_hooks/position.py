"""Position re-encoding strategies for KV cache injection.

Strategy pattern: different model families use different position encoding
schemes. The PositionEncoder protocol lets the injection layer handle any
model's encoding without modification.

- AbsolutePositionEncoder: For models like GPT-2 that use learned absolute
  position embeddings. K vectors don't encode position internally, so
  injection only needs to offset position_ids.

- RoPEPositionEncoder: For models like Llama/Qwen that apply Rotary Position
  Embeddings directly to Q and K vectors. Historical K vectors carry their
  old rotation and must be unrotated then re-rotated at the new position.
"""

from abc import ABC, abstractmethod

import torch


class PositionEncoder(ABC):
    """Abstract strategy for remapping position information during KV injection.

    Different model architectures encode position differently. This protocol
    lets the MemoryInjector handle any model family without modification.
    """

    @abstractmethod
    def remap_keys(
        self,
        keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_start: int,
    ) -> torch.Tensor:
        """Remap position encoding in key tensors.

        Args:
            keys: Key tensor, shape (batch, heads, seq, head_dim).
            old_positions: Original position IDs when these keys were computed.
            new_start: Starting position ID in the current context.

        Returns:
            Keys with position encoding remapped to new_start..new_start+seq.
        """

    @abstractmethod
    def build_position_ids(
        self,
        cache_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Build position_ids tensor accounting for injected cache length.

        Args:
            cache_len: Number of KV entries in the injected cache.
            seq_len: Number of new input tokens.

        Returns:
            position_ids tensor, shape (1, seq_len).
        """


class AbsolutePositionEncoder(PositionEncoder):
    """Position strategy for models with learned absolute position embeddings (GPT-2).

    K vectors don't encode position internally — position is added to
    the input before Q/K/V projection. Injection only requires offsetting
    position_ids so the model knows where the new tokens sit relative to
    the injected cache.
    """

    def remap_keys(
        self,
        keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_start: int,
    ) -> torch.Tensor:
        """No-op: absolute position models don't encode position in K vectors."""
        return keys

    def build_position_ids(
        self,
        cache_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Offset position IDs past the injected cache entries."""
        return torch.arange(cache_len, cache_len + seq_len, dtype=torch.long).unsqueeze(0)


class RoPEPositionEncoder(PositionEncoder):
    """Position strategy for models with Rotary Position Embeddings (Llama, Qwen).

    RoPE applies a position-dependent rotation to Q and K vectors after
    projection. Historical K vectors carry rotation at their original
    position. To inject them at a new position:
      1. Unrotate: undo the rotation at old_position
      2. Re-rotate: apply rotation at new_position

    The rotation operates on consecutive pairs of dimensions:
      [x0, x1] → [x0·cos(θ) - x1·sin(θ), x0·sin(θ) + x1·cos(θ)]
    where θ = position / (10000^(2i/d)) for dimension pair i.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base

    def _compute_freqs(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos frequency tables for given positions."""
        dim_pairs = self.head_dim // 2
        freq_exponents = torch.arange(0, dim_pairs, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (freq_exponents / dim_pairs))

        # positions: (seq,) → angles: (seq, dim_pairs)
        angles = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to a key tensor.

        Args:
            x: shape (batch, heads, seq, head_dim)
            cos, sin: shape (seq, head_dim // 2)
        """
        x0 = x[..., 0::2]  # even dims
        x1 = x[..., 1::2]  # odd dims

        # Broadcast cos/sin to (1, 1, seq, head_dim//2).
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_even = x0 * cos - x1 * sin
        rotated_odd = x0 * sin + x1 * cos

        # Interleave back to (batch, heads, seq, head_dim).
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)

    @staticmethod
    def _unrotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Undo RoPE rotation (inverse = rotate with -sin)."""
        return RoPEPositionEncoder._rotate(x, cos, -sin)

    def remap_keys(
        self,
        keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_start: int,
    ) -> torch.Tensor:
        """Unrotate keys at old positions, re-rotate at new positions."""
        seq_len = keys.shape[2]

        # Unrotate at old positions.
        old_cos, old_sin = self._compute_freqs(old_positions)
        keys = self._unrotate(keys, old_cos, old_sin)

        # Re-rotate at new positions.
        new_positions = torch.arange(new_start, new_start + seq_len)
        new_cos, new_sin = self._compute_freqs(new_positions)
        return self._rotate(keys, new_cos, new_sin)

    def build_position_ids(
        self,
        cache_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Offset position IDs past the injected cache entries."""
        return torch.arange(cache_len, cache_len + seq_len, dtype=torch.long).unsqueeze(0)
