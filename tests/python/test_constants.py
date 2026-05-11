"""Acceptance tests for the shared constants module.

Ensures every named constant used by file-ingest and multi-view
consolidation features is exported from a single source of truth.
"""

from tardigrade_hooks.constants import (
    EDGE_CAUSED_BY,
    EDGE_CONTRADICTS,
    EDGE_FOLLOWS,
    EDGE_SUPPORTS,
    CONSOLIDATION_BATCH_SIZE,
    CONSOLIDATION_MIN_TIER,
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_CAPTURE_LAYER_RATIO,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_FILE_INGEST_SALIENCE,
    DEFAULT_VIEW_FRAMINGS,
    MAX_VIEWS_PER_MEMORY,
    MIN_CHUNK_TOKENS,
    VIEW_FRAMING_PARAPHRASE,
    VIEW_FRAMING_QUESTION,
    VIEW_FRAMING_SUMMARY,
)


class TestEdgeTypes:
    """Edge type constants match the Rust EdgeType enum in tdb-engine."""

    def test_edge_values_match_rust_enum(self):
        assert EDGE_CAUSED_BY == 0
        assert EDGE_FOLLOWS == 1
        assert EDGE_CONTRADICTS == 2
        assert EDGE_SUPPORTS == 3

    def test_edge_types_are_distinct(self):
        values = [EDGE_CAUSED_BY, EDGE_FOLLOWS, EDGE_CONTRADICTS, EDGE_SUPPORTS]
        assert len(set(values)) == 4


class TestLayerSelection:
    def test_capture_layer_ratio_is_two_thirds(self):
        assert DEFAULT_CAPTURE_LAYER_RATIO == 0.67

    def test_capture_layer_for_28_layer_model(self):
        n_layers = 28
        assert int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO) == 18


class TestChunkingDefaults:
    def test_chunk_tokens_positive(self):
        assert DEFAULT_CHUNK_TOKENS > 0

    def test_overlap_less_than_chunk(self):
        assert CHUNK_OVERLAP_TOKENS < DEFAULT_CHUNK_TOKENS

    def test_min_chunk_positive(self):
        assert MIN_CHUNK_TOKENS > 0
        assert MIN_CHUNK_TOKENS < DEFAULT_CHUNK_TOKENS


class TestMultiViewDefaults:
    def test_max_views_positive(self):
        assert MAX_VIEWS_PER_MEMORY >= 1

    def test_default_framings_are_tuple(self):
        assert isinstance(DEFAULT_VIEW_FRAMINGS, tuple)

    def test_default_framings_contain_three_types(self):
        assert len(DEFAULT_VIEW_FRAMINGS) == 3
        assert VIEW_FRAMING_SUMMARY in DEFAULT_VIEW_FRAMINGS
        assert VIEW_FRAMING_QUESTION in DEFAULT_VIEW_FRAMINGS
        assert VIEW_FRAMING_PARAPHRASE in DEFAULT_VIEW_FRAMINGS


class TestConsolidationDefaults:
    def test_min_tier_is_validated(self):
        assert CONSOLIDATION_MIN_TIER == 1

    def test_batch_size_positive(self):
        assert CONSOLIDATION_BATCH_SIZE > 0


class TestFileSalience:
    def test_salience_in_valid_range(self):
        assert 0.0 < DEFAULT_FILE_INGEST_SALIENCE <= 100.0
