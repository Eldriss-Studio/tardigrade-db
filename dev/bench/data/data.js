window.BENCHMARK_DATA = {
  "lastUpdate": 1776884821232,
  "repoUrl": "https://github.com/Eldriss-Studio/tardigrade-db",
  "entries": {
    "TardigradeDB Performance": [
      {
        "commit": {
          "author": {
            "email": "flagrare@live.it",
            "name": "Flagrare",
            "username": "Flagrare"
          },
          "committer": {
            "email": "flagrare@live.it",
            "name": "Flagrare",
            "username": "Flagrare"
          },
          "distinct": true,
          "id": "bbe164c07729327e63e51c50010d521df42a0453",
          "message": "🐛 fix(ci): run each criterion bench target individually (again)\n\nThe workspace-level cargo bench passes --output-format bencher to lib\ntest binaries that don't support it. Must run each --bench target by\nname. Previous fix was overwritten when bench.yml was modified.",
          "timestamp": "2026-04-22T15:56:38-03:00",
          "tree_id": "d611f0e1ad8486c88ff1cc7308fbfc7e4a53dfdf",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/bbe164c07729327e63e51c50010d521df42a0453"
        },
        "date": 1776884820441,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 278,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 531,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1077,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2084,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4161,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 97,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 185,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 350,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 707,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1366,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 715,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1418,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2813,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 333840,
            "range": "± 46893",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 17850,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/64",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/128",
            "value": 95,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/256",
            "value": 215,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 455,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/64",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/128",
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/256",
            "value": 64,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/512",
            "value": 124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/256",
            "value": 10848,
            "range": "± 269",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 41304,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 163650,
            "range": "± 339",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 291669,
            "range": "± 2045",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8340990,
            "range": "± 22368",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 34054728,
            "range": "± 72691",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 14427,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 29898845,
            "range": "± 1817940",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 304434255,
            "range": "± 9064125",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3027672,
            "range": "± 90147",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 330764,
            "range": "± 63505",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 114981,
            "range": "± 1983",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}