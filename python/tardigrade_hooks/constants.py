"""Shared named constants for TardigradeDB Python features.

Single source of truth for tunable values used across file ingestion,
multi-view consolidation, and governance.  Values mirror the Rust-side
``EdgeType`` enum in ``tdb-engine`` where applicable.
"""

# -- Edge types (must match Rust EdgeType discriminants) ---------------------
EDGE_CAUSED_BY: int = 0
EDGE_FOLLOWS: int = 1
EDGE_CONTRADICTS: int = 2
EDGE_SUPPORTS: int = 3

# -- Layer selection ---------------------------------------------------------
DEFAULT_CAPTURE_LAYER_RATIO: float = 0.67

# -- Chunking (file ingestion) ----------------------------------------------
DEFAULT_CHUNK_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
MIN_CHUNK_TOKENS: int = 32

# -- Multi-view consolidation ------------------------------------------------
MAX_VIEWS_PER_MEMORY: int = 4
VIEW_FRAMING_SUMMARY: str = "summary"
VIEW_FRAMING_QUESTION: str = "question"
VIEW_FRAMING_PARAPHRASE: str = "paraphrase"
DEFAULT_VIEW_FRAMINGS: tuple[str, ...] = (
    VIEW_FRAMING_SUMMARY,
    VIEW_FRAMING_QUESTION,
    VIEW_FRAMING_PARAPHRASE,
)

# -- Consolidation scheduling ------------------------------------------------
CONSOLIDATION_MIN_TIER: int = 1  # Validated tier
CONSOLIDATION_BATCH_SIZE: int = 16

# -- File ingest salience ----------------------------------------------------
DEFAULT_FILE_INGEST_SALIENCE: float = 70.0

# -- Multi-view v2: LLM generation + diversity filter ----------------------
VIEW_DIVERSITY_THRESHOLD: float = 0.92
MAX_VIEW_CANDIDATES: int = 5
MAX_VIEWS_KEPT: int = 3
LLM_VIEW_PROMPT_TEMPLATE: str = (
    "Write one specific question that the following fact can answer. "
    "Use different words than the original fact.\n\n"
    "Fact: {fact_text}\n\nQuestion:"
)

# -- RLS (Reflective Latent Search) -----------------------------------------
RLS_MODE_NONE: str = "none"
RLS_MODE_KEYWORD: str = "keyword"
RLS_MODE_MULTIPHRASING: str = "multiphrasing"
RLS_MODE_EMBEDDING: str = "embedding"
RLS_MODE_GENERATIVE: str = "generative"
RLS_MODE_BOTH: str = "both"
RLS_DEFAULT_CONFIDENCE_THRESHOLD: float = 1.5
RLS_DEFAULT_MAX_ATTEMPTS: int = 2
RLS_DEFAULT_GEN_MODEL: str = "Qwen/Qwen2.5-3B"
RLS_REFORMULATION_PROMPT: str = (
    'Rephrase this question using different words:\n'
    '"{query_text}"\n'
    'Rephrased:'
)
