# Per-token encoding — shared header format for TardigradeDB.
#
# Constants are defined in Rust (tdb-retrieval/src/per_token.rs) and
# re-exported via the tardigrade_db Python module. This file resolves
# them lazily via PEP 562 module __getattr__ so `from tardigrade_hooks
# import …` (or any other path that touches this module's import) does
# NOT eagerly require the compiled native extension. CI lint jobs that
# don't run `maturin develop` would otherwise break with
# `ModuleNotFoundError: No module named 'tardigrade_db._native'`.
#
# Header layout (64 floats = two Q4 groups):
#   Group 0 (indices 0-31):  sentinel (-1e9) + zeros
#   Group 1 (indices 32-63): n_tokens + dim + zeros
#   Data (index 64+):        concatenated per-token vectors

import numpy as np

# SENTINEL_IDX is pure Python, no native required.
SENTINEL_IDX: int = 0


_LAZY_ATTRS = {
    "HEADER_SIZE",
    "SENTINEL_VALUE",
    "N_TOKENS_IDX",
    "DIM_IDX",
}


def __getattr__(name: str):
    """Lazy module attribute resolution for native-defined constants.

    Importing :mod:`tardigrade_db` at module load would break CI lint
    paths that import sibling modules of :mod:`tardigrade_hooks`
    without first building the native extension. PEP 562 lets us
    defer the import until a consumer actually reads one of the
    native-derived constants.
    """
    if name in _LAZY_ATTRS:
        import tardigrade_db

        if name == "HEADER_SIZE":
            return tardigrade_db.ENCODING_HEADER_SIZE
        if name == "SENTINEL_VALUE":
            return tardigrade_db.ENCODING_SENTINEL
        if name == "N_TOKENS_IDX":
            return tardigrade_db.ENCODING_N_TOKENS_IDX
        if name == "DIM_IDX":
            return tardigrade_db.ENCODING_DIM_IDX
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def encode_per_token(token_vecs, dim):
    """Encode per-token vectors with Q4-safe sentinel header.

    Only ``dim`` is stored in the header — it is the abs_max of its Q4 group
    and survives quantization exactly. ``n_tokens`` is left as 0.0 because Q4
    would corrupt it; readers compute ``n = data.len() / dim`` instead. See
    ``tdb-retrieval/per_token.rs::HEADER_SIZE`` for the full contract.

    Args:
        token_vecs: numpy array of shape (n_tokens, dim).
        dim: dimension of each token vector.

    Returns:
        numpy array: [64-byte header | flattened token vectors]
    """
    import tardigrade_db  # local: lazy load to keep module import light

    header = np.zeros(tardigrade_db.ENCODING_HEADER_SIZE, dtype=np.float32)
    header[SENTINEL_IDX] = tardigrade_db.ENCODING_SENTINEL
    header[tardigrade_db.ENCODING_DIM_IDX] = float(dim)
    return np.concatenate([header, token_vecs.ravel()])
