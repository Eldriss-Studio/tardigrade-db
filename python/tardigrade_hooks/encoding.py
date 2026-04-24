# Per-token encoding — shared header format for TardigradeDB.
#
# Single source of truth for the Q4-safe sentinel header encoding.
# Used by both HuggingFaceKVHook and KnowledgePackStore.
#
# Header layout (64 floats = two Q4 groups):
#   Group 0 (indices 0-31):  sentinel (-1e9) + zeros
#   Group 1 (indices 32-63): n_tokens + dim + zeros
#   Data (index 64+):        concatenated per-token vectors
#
# The sentinel and metadata are in separate Q4 groups so the
# sentinel's magnitude doesn't crush the metadata during Q4
# quantization (where scale = max(abs) / 7 per 32-value group).

import numpy as np

HEADER_SIZE = 64
SENTINEL_VALUE = -1.0e9
SENTINEL_IDX = 0
N_TOKENS_IDX = 32
DIM_IDX = 33


def encode_per_token(token_vecs, dim):
    """Encode per-token vectors with Q4-safe sentinel header.

    Args:
        token_vecs: numpy array of shape (n_tokens, dim).
        dim: dimension of each token vector.

    Returns:
        numpy array: [64-byte header | flattened token vectors]
    """
    n = len(token_vecs)
    header = np.zeros(HEADER_SIZE, dtype=np.float32)
    header[SENTINEL_IDX] = SENTINEL_VALUE
    header[N_TOKENS_IDX] = float(n)
    header[DIM_IDX] = float(dim)
    return np.concatenate([header, token_vecs.ravel()])
