window.BENCHMARK_DATA = {
  "lastUpdate": 1778483198403,
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
      },
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
          "id": "ba12e2ac9cde90ba3cb7e48561f4a0e6faa01e27",
          "message": "🐛 fix(docs): correct API docs link in bench dashboard (../../ not ../)",
          "timestamp": "2026-04-22T16:09:22-03:00",
          "tree_id": "845b25f45a6145206c9d91ae64145b43fe792826",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/ba12e2ac9cde90ba3cb7e48561f4a0e6faa01e27"
        },
        "date": 1776885564730,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 277,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 529,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1070,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2091,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4153,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 96,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 186,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 349,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 707,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1363,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 715,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1418,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2814,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 352439,
            "range": "± 29613",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 17371,
            "range": "± 40",
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
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 454,
            "range": "± 0",
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
            "value": 65,
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
            "value": 10819,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 41379,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 162773,
            "range": "± 353",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 292007,
            "range": "± 11321",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8089653,
            "range": "± 6793",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 32899480,
            "range": "± 49362",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 14413,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 44061369,
            "range": "± 5875280",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 459207921,
            "range": "± 66443416",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3013460,
            "range": "± 17700",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 482821,
            "range": "± 4845442",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 114655,
            "range": "± 335",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "committer": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "id": "4cd33b1fba69fd7eecb56d321fba7b65321584d2",
          "message": "ci(bench): make full compare/publish resilient to placeholder baselines",
          "timestamp": "2026-04-22T23:52:24Z",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/4cd33b1fba69fd7eecb56d321fba7b65321584d2"
        },
        "date": 1776902072355,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 226,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 432,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 845,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 1678,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 3374,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 81,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 153,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 294,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 612,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1178,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 584,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1139,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2286,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 343210,
            "range": "± 1526517",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 16460,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/64",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/128",
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/256",
            "value": 180,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/64",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/128",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/256",
            "value": 53,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/512",
            "value": 99,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/256",
            "value": 8427,
            "range": "± 181",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 32105,
            "range": "± 270",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 126923,
            "range": "± 1816",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 216227,
            "range": "± 308",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 6257664,
            "range": "± 44021",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 25980412,
            "range": "± 204931",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 11606,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 37465362,
            "range": "± 67970592",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 651632540,
            "range": "± 559683836",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 2819885,
            "range": "± 7861",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 330027,
            "range": "± 1281485",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 99388,
            "range": "± 447",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "committer": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "id": "58079e622f4034683abfb78f05f84af051798351",
          "message": "✨ feat(vllm): align retrieval key with last-token-of-last-layer (Step 6)\n\nReplace the mean-pooled layer-0 K retrieval key with the last-token K\nprojection from the last layer. Mirror the change on the query side: use\nthe last token's embedding instead of the mean.\n\nWhy:\n- Engine evidence (docs/experiments/kv-cache-validation.md): mean-pool\n  produces a 31% recall \"gravity well\" — averaged vectors land in a dense\n  region near every other averaged vector, so dot products fail to\n  discriminate. Last-token data carries the model's \"current focus\"\n  after attending over the prompt, which is a much sharper signal.\n- Asymmetric save/query strategies match: save uses last-token K from\n  the deepest layer (post-attention); query uses last-token embedding\n  (cheap CPU lookup, no GPU forward). Aligned positionally even though\n  the source representations differ.\n- Constraint: mem_write_pack accepts one key per pack. The richer\n  per-token Top5Avg path that the engine validated to 100% recall\n  needs a different storage primitive (per-cell write); deferred.\n- Constraint: requires hidden_size == kv_dim, which holds for Qwen3-0.6B\n  (1024 == 8*128). For models where they differ, a future projection\n  layer is needed.\n\nResult on the A/B test (test_primed_request_changes_generation_after_relevant_save):\n- Still SKIPs. Diagnostic logging revealed the gating issue is upstream:\n  the scheduler-side connector instance sees pack_count=0 for the entire\n  test session because TardigradeDB engine state is cached at open-time\n  and the scheduler never sees the worker-side writes.\n- This is Gap 7 in the original gap-closure plan: engine cross-process\n  visibility. Documented as a separate Rust engine RFC. Until that lands,\n  Step 6's improved keys are stored correctly but never queried against\n  on the matching path.\n\nThe connector-side change is correct in isolation. Verified by full suite:\n25 passed + 1 skipped (the A/B), 0 regressions on save / load / spy /\ncross-session / format tests.",
          "timestamp": "2026-04-27T02:10:45Z",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/58079e622f4034683abfb78f05f84af051798351"
        },
        "date": 1777271392089,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 281,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 534,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1072,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2102,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4160,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 102,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 185,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 349,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 710,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1449,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 720,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1421,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2812,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 392195,
            "range": "± 48711",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 17464,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-100/100",
            "value": 17276,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-100/100",
            "value": 17454,
            "range": "± 82",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-1000/1000",
            "value": 17317,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-1000/1000",
            "value": 17522,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-10000/10000",
            "value": 17460,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-10000/10000",
            "value": 17647,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/64",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/128",
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/256",
            "value": 207,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 447,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/64",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/128",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/256",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/512",
            "value": 122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/256",
            "value": 10877,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 41349,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 163460,
            "range": "± 288",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-100/100",
            "value": 359349,
            "range": "± 990",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/1000",
            "value": 1481158,
            "range": "± 11864",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/10000",
            "value": 4284358,
            "range": "± 184571",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 291599,
            "range": "± 723",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8348459,
            "range": "± 49611",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 33093956,
            "range": "± 115864",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 14714,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 36765695,
            "range": "± 3877129",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 364711166,
            "range": "± 24008899",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3020011,
            "range": "± 9526",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 408766,
            "range": "± 4846060",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 114021,
            "range": "± 920",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-100-vamana-changed-...",
            "value": 467581,
            "range": "± 1765",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-...",
            "value": 4271831,
            "range": "± 18322",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-... #2",
            "value": 8070889,
            "range": "± 172568",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-...",
            "value": 367930,
            "range": "± 1698",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-...",
            "value": 367903,
            "range": "± 1434",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-...",
            "value": 468933,
            "range": "± 2837",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-...",
            "value": 470691,
            "range": "± 8008",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-...",
            "value": 757665,
            "range": "± 4327",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-...",
            "value": 764894,
            "range": "± 1798",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128...",
            "value": 1911505,
            "range": "± 20787",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256...",
            "value": 1924509,
            "range": "± 22180",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #2",
            "value": 4158618,
            "range": "± 19763",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #2",
            "value": 4170980,
            "range": "± 13907",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #2",
            "value": 4262609,
            "range": "± 16840",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #2",
            "value": 4282580,
            "range": "± 17925",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #2",
            "value": 4556177,
            "range": "± 15317",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #2",
            "value": 4565157,
            "range": "± 9728",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #2",
            "value": 5723641,
            "range": "± 11610",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #2",
            "value": 5736546,
            "range": "± 22300",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #3",
            "value": 13047053,
            "range": "± 204477",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #3",
            "value": 13040302,
            "range": "± 257966",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #3",
            "value": 13262832,
            "range": "± 253461",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #3",
            "value": 13174146,
            "range": "± 272127",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #3",
            "value": 13584561,
            "range": "± 265325",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #3",
            "value": 13580019,
            "range": "± 299624",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #3",
            "value": 14901673,
            "range": "± 343279",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #3",
            "value": 14769972,
            "range": "± 299074",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind...",
            "value": 459556,
            "range": "± 1200",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind...",
            "value": 458438,
            "range": "± 9369",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind...",
            "value": 459817,
            "range": "± 2572",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind...",
            "value": 458745,
            "range": "± 1855",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind...",
            "value": 460796,
            "range": "± 1998",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind...",
            "value": 460936,
            "range": "± 1237",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in...",
            "value": 461295,
            "range": "± 955",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in...",
            "value": 461842,
            "range": "± 5810",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #2",
            "value": 4247833,
            "range": "± 17871",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #2",
            "value": 4253874,
            "range": "± 17231",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #2",
            "value": 4250395,
            "range": "± 7794",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #2",
            "value": 4255026,
            "range": "± 14796",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #2",
            "value": 4291607,
            "range": "± 10449",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #2",
            "value": 4262881,
            "range": "± 119362",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #2",
            "value": 4281415,
            "range": "± 20002",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #2",
            "value": 4277513,
            "range": "± 8977",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #3",
            "value": 8052955,
            "range": "± 246756",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #3",
            "value": 7941976,
            "range": "± 133882",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #3",
            "value": 8121298,
            "range": "± 410692",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #3",
            "value": 8034420,
            "range": "± 259155",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #3",
            "value": 8085886,
            "range": "± 201554",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #3",
            "value": 7969635,
            "range": "± 152126",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #3",
            "value": 8018772,
            "range": "± 148860",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #3",
            "value": 8016266,
            "range": "± 233293",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "committer": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "id": "c7963cdd164aaeb1783070f3f1fff0136af968b2",
          "message": "✨ feat(reranker): cross-encoder Stage-2 — vague R@5 46% → 64%, moderate 28% → 68%\n\nAdds python/tardigrade_hooks/reranker.py::CrossEncoderReranker, a Stage-2\ntext reranker that runs over the engine's top-K candidates when memo text\nis available. Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M\nparams, MiniLM arch arXiv:2002.10957, trained on MS MARCO arXiv:1611.09268,\nwrapped via sentence-transformers arXiv:1908.10084).\n\nWhy this works where PRF didn't: the cross-encoder lets every query token\nattend to every document token jointly — strictly more expressive than\nthe bi-encoder dot product. PRF tried to perturb the query indirectly;\nthe cross-encoder does the right thing directly with a tiny dedicated\nmodel.\n\nEmpirical (RTX 3070 Ti, Qwen3-0.6B, 100-cell corpus, 230 queries):\n\n| mode              | specific | moderate    | vague       | p95   |\n|-------------------|----------|-------------|-------------|-------|\n| baseline (none)   | 100%     | 28%         | 46%         | 67ms  |\n| centered          | 100%     | 59% (+31pp) | 50% (+4pp)  | 58ms  |\n| none + rerank     | 100%     | 57% (+29pp) | 62% (+16pp) | 77ms  |\n| centered + rerank | 100%     | 68% (+40pp) | 64% (+18pp) | 86ms  |\n\nStacking is additive — mean-centering improves the candidate set the\nreranker scores from. The cross-encoder gets the best of both: better\ncandidates AND stronger pairwise scoring.\n\nAPI:\n  from tardigrade_hooks.reranker import CrossEncoderReranker\n  reranker = CrossEncoderReranker()\n  ordered = reranker.rerank(\n      query_text=question,\n      candidates=engine.mem_read_tokens(query_tokens, k=10, owner=1),\n      get_text=lambda h: text_lookup[int(h.cell_id)],\n  )\n\nBench harness integration:\n- TDB_BENCH_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 in the\n  Tardigrade adapter enables Stage-2 reranking transparently.\n- docs/experiments/vague_queries/bench_vague.py grew --rerank flag.\n\nCaveat: MS MARCO is short web passages, not first-person diary text.\nFine-tuning the reranker on agent-memory data would likely give another\n5–10pp. Tracked as future work.\n\nTests: +5 Python (test_reranker.py). Full suite green: 300 Rust + 279\nPython = 579 total.\n\nExternal references added to docs/refs/external-references.md:\n- MiniLM (arXiv:2002.10957) in B1\n- Sentence-BERT/sentence-transformers (arXiv:1908.10084) in B1\n- MS MARCO (arXiv:1611.09268) in B1\n- Pinecone two-stage retrieval pattern in B1\n- Cross-encoder reranker in C1 implementation table\n- Mean-centering + Latent PRF rows added to C1 (algorithms implemented)",
          "timestamp": "2026-05-02T22:56:21Z",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/c7963cdd164aaeb1783070f3f1fff0136af968b2"
        },
        "date": 1777876842830,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 301,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 573,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1100,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2169,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4413,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 111,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 202,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 385,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 790,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1524,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 754,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1469,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2999,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 217352,
            "range": "± 11812",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 21651,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-100/100",
            "value": 20470,
            "range": "± 272",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-100/100",
            "value": 21541,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-1000/1000",
            "value": 21230,
            "range": "± 160",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-1000/1000",
            "value": 21453,
            "range": "± 88",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-10000/10000",
            "value": 21770,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-10000/10000",
            "value": 21879,
            "range": "± 47",
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
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/256",
            "value": 219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 489,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/64",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/128",
            "value": 6,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/256",
            "value": 9,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/512",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/256",
            "value": 3279,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 10888,
            "range": "± 92",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 40006,
            "range": "± 725",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-100/100",
            "value": 138932,
            "range": "± 294",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/1000",
            "value": 683099,
            "range": "± 4037",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/10000",
            "value": 3468007,
            "range": "± 11317",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 281527,
            "range": "± 266",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8329043,
            "range": "± 28269",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 32900070,
            "range": "± 97867",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 15007,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 21053670,
            "range": "± 1021922",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 210541959,
            "range": "± 7976615",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3734311,
            "range": "± 7095",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 241985,
            "range": "± 80537",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 218543,
            "range": "± 513",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-100-vamana-changed-...",
            "value": 367917,
            "range": "± 1002",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-...",
            "value": 1940620,
            "range": "± 7737",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-... #2",
            "value": 5270460,
            "range": "± 47707",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-...",
            "value": 144139,
            "range": "± 581",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-...",
            "value": 141062,
            "range": "± 541",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-...",
            "value": 257808,
            "range": "± 798",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-...",
            "value": 261608,
            "range": "± 826",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-...",
            "value": 606773,
            "range": "± 1312",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-...",
            "value": 606174,
            "range": "± 5485",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128...",
            "value": 1981696,
            "range": "± 3602",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256...",
            "value": 1995123,
            "range": "± 6784",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #2",
            "value": 1679456,
            "range": "± 18963",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #2",
            "value": 1673898,
            "range": "± 6645",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #2",
            "value": 1804554,
            "range": "± 17206",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #2",
            "value": 1812603,
            "range": "± 7791",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #2",
            "value": 2154411,
            "range": "± 19283",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #2",
            "value": 2166047,
            "range": "± 6786",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #2",
            "value": 3502578,
            "range": "± 8646",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #2",
            "value": 3519295,
            "range": "± 20775",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #3",
            "value": 7498664,
            "range": "± 198262",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #3",
            "value": 7136612,
            "range": "± 145188",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #3",
            "value": 7544161,
            "range": "± 573765",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #3",
            "value": 8749089,
            "range": "± 118912",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #3",
            "value": 9214039,
            "range": "± 575537",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #3",
            "value": 9176996,
            "range": "± 137746",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #3",
            "value": 10824567,
            "range": "± 131774",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #3",
            "value": 10898211,
            "range": "± 135930",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind...",
            "value": 356776,
            "range": "± 1145",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind...",
            "value": 354426,
            "range": "± 910",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind...",
            "value": 349267,
            "range": "± 1438",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind...",
            "value": 354993,
            "range": "± 1168",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind...",
            "value": 354734,
            "range": "± 1426",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind...",
            "value": 353733,
            "range": "± 3464",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in...",
            "value": 355288,
            "range": "± 2360",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in...",
            "value": 355594,
            "range": "± 1294",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #2",
            "value": 1905504,
            "range": "± 10777",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #2",
            "value": 1924766,
            "range": "± 12175",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #2",
            "value": 1921392,
            "range": "± 28243",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #2",
            "value": 1908867,
            "range": "± 36745",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #2",
            "value": 1924234,
            "range": "± 16393",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #2",
            "value": 1931925,
            "range": "± 12452",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #2",
            "value": 1912158,
            "range": "± 38576",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #2",
            "value": 1929594,
            "range": "± 18485",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #3",
            "value": 6284503,
            "range": "± 175054",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #3",
            "value": 6445073,
            "range": "± 158305",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #3",
            "value": 6503219,
            "range": "± 139387",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #3",
            "value": 6167876,
            "range": "± 221576",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #3",
            "value": 6526412,
            "range": "± 101250",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #3",
            "value": 6449436,
            "range": "± 332516",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #3",
            "value": 5435883,
            "range": "± 199356",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #3",
            "value": 5475046,
            "range": "± 96983",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "committer": {
            "name": "Flagrare",
            "username": "Flagrare",
            "email": "flagrare@live.it"
          },
          "id": "c7963cdd164aaeb1783070f3f1fff0136af968b2",
          "message": "✨ feat(reranker): cross-encoder Stage-2 — vague R@5 46% → 64%, moderate 28% → 68%\n\nAdds python/tardigrade_hooks/reranker.py::CrossEncoderReranker, a Stage-2\ntext reranker that runs over the engine's top-K candidates when memo text\nis available. Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M\nparams, MiniLM arch arXiv:2002.10957, trained on MS MARCO arXiv:1611.09268,\nwrapped via sentence-transformers arXiv:1908.10084).\n\nWhy this works where PRF didn't: the cross-encoder lets every query token\nattend to every document token jointly — strictly more expressive than\nthe bi-encoder dot product. PRF tried to perturb the query indirectly;\nthe cross-encoder does the right thing directly with a tiny dedicated\nmodel.\n\nEmpirical (RTX 3070 Ti, Qwen3-0.6B, 100-cell corpus, 230 queries):\n\n| mode              | specific | moderate    | vague       | p95   |\n|-------------------|----------|-------------|-------------|-------|\n| baseline (none)   | 100%     | 28%         | 46%         | 67ms  |\n| centered          | 100%     | 59% (+31pp) | 50% (+4pp)  | 58ms  |\n| none + rerank     | 100%     | 57% (+29pp) | 62% (+16pp) | 77ms  |\n| centered + rerank | 100%     | 68% (+40pp) | 64% (+18pp) | 86ms  |\n\nStacking is additive — mean-centering improves the candidate set the\nreranker scores from. The cross-encoder gets the best of both: better\ncandidates AND stronger pairwise scoring.\n\nAPI:\n  from tardigrade_hooks.reranker import CrossEncoderReranker\n  reranker = CrossEncoderReranker()\n  ordered = reranker.rerank(\n      query_text=question,\n      candidates=engine.mem_read_tokens(query_tokens, k=10, owner=1),\n      get_text=lambda h: text_lookup[int(h.cell_id)],\n  )\n\nBench harness integration:\n- TDB_BENCH_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 in the\n  Tardigrade adapter enables Stage-2 reranking transparently.\n- docs/experiments/vague_queries/bench_vague.py grew --rerank flag.\n\nCaveat: MS MARCO is short web passages, not first-person diary text.\nFine-tuning the reranker on agent-memory data would likely give another\n5–10pp. Tracked as future work.\n\nTests: +5 Python (test_reranker.py). Full suite green: 300 Rust + 279\nPython = 579 total.\n\nExternal references added to docs/refs/external-references.md:\n- MiniLM (arXiv:2002.10957) in B1\n- Sentence-BERT/sentence-transformers (arXiv:1908.10084) in B1\n- MS MARCO (arXiv:1611.09268) in B1\n- Pinecone two-stage retrieval pattern in B1\n- Cross-encoder reranker in C1 implementation table\n- Mean-centering + Latent PRF rows added to C1 (algorithms implemented)",
          "timestamp": "2026-05-02T22:56:21Z",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/c7963cdd164aaeb1783070f3f1fff0136af968b2"
        },
        "date": 1778483197672,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 311,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/128",
            "value": 566,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1099,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2190,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4327,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 111,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 203,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 384,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 787,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1520,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 755,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1469,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2946,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 212578,
            "range": "± 14260",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 21053,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-100/100",
            "value": 20517,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-100/100",
            "value": 20849,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-1000/1000",
            "value": 20599,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-1000/1000",
            "value": 20907,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-128-cells-10000/10000",
            "value": 21122,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "BlockPool get — storage hydration/payload-256-cells-10000/10000",
            "value": 21386,
            "range": "± 74",
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
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/256",
            "value": 219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 489,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/64",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/128",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/256",
            "value": 9,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "INT8 dot product — NEON-accelerated attention score/dim/512",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/256",
            "value": 3270,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 10877,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 41279,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-100/100",
            "value": 137807,
            "range": "± 560",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/1000",
            "value": 680223,
            "range": "± 4481",
            "unit": "ns/iter"
          },
          {
            "name": "PerTokenRetriever Top5Avg query — encoded per-token keys/cells-r1-100-r5-100-gw-1-cand-320/10000",
            "value": 3496298,
            "range": "± 19168",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 282603,
            "range": "± 171",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8320870,
            "range": "± 54922",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 32916657,
            "range": "± 161945",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 15428,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 20519677,
            "range": "± 914456",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 211122698,
            "range": "± 7324834",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3703773,
            "range": "± 10840",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 242031,
            "range": "± 96571",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 214067,
            "range": "± 703",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-100-vamana-changed-...",
            "value": 368711,
            "range": "± 721",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-...",
            "value": 1941215,
            "range": "± 17260",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — encoded per-token Top5Avg path/cells-r1-100-r5-100-gw-1-cand-320-vamana-changed-... #2",
            "value": 5519090,
            "range": "± 158019",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-...",
            "value": 147492,
            "range": "± 614",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-...",
            "value": 145731,
            "range": "± 2924",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-...",
            "value": 256694,
            "range": "± 4045",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-...",
            "value": 260730,
            "range": "± 1229",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-...",
            "value": 593585,
            "range": "± 6255",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-...",
            "value": 598446,
            "range": "± 2418",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128...",
            "value": 1945394,
            "range": "± 7217",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256...",
            "value": 1957463,
            "range": "± 64005",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #2",
            "value": 1711002,
            "range": "± 9948",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #2",
            "value": 1706523,
            "range": "± 14173",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #2",
            "value": 1819745,
            "range": "± 65476",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #2",
            "value": 1817889,
            "range": "± 18701",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #2",
            "value": 2171524,
            "range": "± 15152",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #2",
            "value": 2171228,
            "range": "± 18756",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #2",
            "value": 3527797,
            "range": "± 17573",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #2",
            "value": 3540653,
            "range": "± 14335",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-128-... #3",
            "value": 7258781,
            "range": "± 243943",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-0-payload-256-... #3",
            "value": 7240908,
            "range": "± 205889",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-128-... #3",
            "value": 7398076,
            "range": "± 205627",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-1-payload-256-... #3",
            "value": 7346917,
            "range": "± 281871",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-128-... #3",
            "value": 7894225,
            "range": "± 258914",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-4-payload-256-... #3",
            "value": 7836552,
            "range": "± 287227",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-128... #3",
            "value": 9302156,
            "range": "± 408619",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack — encoded per-token Top5Avg path/target-true-dedup-true-layers-16-payload-256... #3",
            "value": 9443449,
            "range": "± 419206",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind...",
            "value": 352244,
            "range": "± 1788",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind...",
            "value": 352040,
            "range": "± 1435",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind...",
            "value": 353068,
            "range": "± 5808",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind...",
            "value": 355224,
            "range": "± 1156",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind...",
            "value": 351522,
            "range": "± 1487",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind...",
            "value": 352157,
            "range": "± 1631",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in...",
            "value": 349894,
            "range": "± 1288",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in...",
            "value": 351994,
            "range": "± 1192",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #2",
            "value": 1926426,
            "range": "± 23916",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #2",
            "value": 1911223,
            "range": "± 29082",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #2",
            "value": 1928731,
            "range": "± 47607",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #2",
            "value": 1924128,
            "range": "± 28583",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #2",
            "value": 1932389,
            "range": "± 29730",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #2",
            "value": 1933071,
            "range": "± 11829",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #2",
            "value": 1940127,
            "range": "± 23567",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #2",
            "value": 1971038,
            "range": "± 109862",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-128-ind... #3",
            "value": 6619515,
            "range": "± 106148",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-0-payload-256-ind... #3",
            "value": 6279503,
            "range": "± 142015",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-128-ind... #3",
            "value": 6675086,
            "range": "± 108257",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-1-payload-256-ind... #3",
            "value": 6362921,
            "range": "± 245092",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-128-ind... #3",
            "value": 6741612,
            "range": "± 547628",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-4-payload-256-ind... #3",
            "value": 6714187,
            "range": "± 137771",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-128-in... #3",
            "value": 6965896,
            "range": "± 385563",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read_pack profile — retrieval cell only/target-true-dedup-true-layers-16-payload-256-in... #3",
            "value": 6218299,
            "range": "± 635981",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}