# Per-token encoding — shared header format for TardigradeDB.
#
# Constants are defined in Rust (tdb-retrieval/src/per_token.rs) and
# re-exported via the tardigrade_db Python module. This file imports
# them so Python callers have a single import path.
#
# Header layout (64 floats = two Q4 groups):
#   Group 0 (indices 0-31):  sentinel (-1e9) + zeros
#   Group 1 (indices 32-63): n_tokens + dim + zeros
#   Data (index 64+):        concatenated per-token vectors

import numpy as np

import tardigrade_db

HEADER_SIZE = tardigrade_db.ENCODING_HEADER_SIZE
SENTINEL_VALUE = tardigrade_db.ENCODING_SENTINEL
SENTINEL_IDX = 0
N_TOKENS_IDX = tardigrade_db.ENCODING_N_TOKENS_IDX
DIM_IDX = tardigrade_db.ENCODING_DIM_IDX


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
