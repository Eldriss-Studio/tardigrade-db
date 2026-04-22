window.BENCHMARK_DATA = {
  "lastUpdate": 1776883610848,
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
          "id": "941d5d4d4503e7188193b28cb14b45d555da2a63",
          "message": "🔄 ci: trigger bench workflow (gh-pages branch now exists)",
          "timestamp": "2026-04-22T15:21:56-03:00",
          "tree_id": "4ece342185115fbef1beae21d8422d51b1042054",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/941d5d4d4503e7188193b28cb14b45d555da2a63"
        },
        "date": 1776882730335,
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
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/256",
            "value": 1077,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2102,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4172,
            "range": "± 10",
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
            "value": 187,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 709,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1452,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 715,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1418,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2801,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 303178,
            "range": "± 22532",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 17457,
            "range": "± 242",
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
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "FP32 dot product — baseline attention score/dim/512",
            "value": 454,
            "range": "± 1",
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
            "value": 10832,
            "range": "± 238",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 41328,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 163283,
            "range": "± 334",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 292014,
            "range": "± 129",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8146612,
            "range": "± 7875",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 33977276,
            "range": "± 99534",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 14442,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 30262196,
            "range": "± 1930666",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 307124715,
            "range": "± 17926546",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3054100,
            "range": "± 16115",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 340981,
            "range": "± 72745",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 115581,
            "range": "± 667",
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
          "id": "cd20939e8d32262100e45f2ba710c8d2f966c594",
          "message": "🔧 fix(ci): deploy docs to gh-pages branch, unifying with bench dashboard\n\nThe doc job was deploying via the Pages artifact API while the bench job\npushed to the gh-pages branch. Pages only served the artifact, making\nthe bench dashboard at /dev/bench/ return 404.\n\nFix: switch doc deployment to peaceiris/actions-gh-pages (push to branch)\nwith keep_files: true (preserves dev/bench/ data from bench workflow).\nConfigure Pages source to deploy from gh-pages branch.\n\nBoth rustdoc and benchmark charts now served from the same branch.",
          "timestamp": "2026-04-22T15:37:13-03:00",
          "tree_id": "fbe984bad6540e8fc651397ed5e947db20cca2aa",
          "url": "https://github.com/Eldriss-Studio/tardigrade-db/commit/cd20939e8d32262100e45f2ba710c8d2f966c594"
        },
        "date": 1776883610467,
        "tool": "cargo",
        "benches": [
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/64",
            "value": 277,
            "range": "± 1",
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
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/512",
            "value": 2099,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 quantize — compress f32 → 4-bit (GGML Q4_0)/floats/1024",
            "value": 4162,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/64",
            "value": 101,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/128",
            "value": 185,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/256",
            "value": 350,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/512",
            "value": 712,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 dequantize — decompress 4-bit → f32/floats/1024",
            "value": 1368,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/128",
            "value": 716,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/256",
            "value": 1420,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Q4 round-trip — quantize + dequantize end-to-end/floats/512",
            "value": 2819,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
            "value": 304655,
            "range": "± 11560",
            "unit": "ns/iter"
          },
          {
            "name": "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
            "value": 17708,
            "range": "± 231",
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
            "value": 96,
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
            "value": 455,
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
            "value": 64,
            "range": "± 1",
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
            "value": 10838,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/1024",
            "value": 41286,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "SLB query — hot-path INT8 cache lookup (top-5)/entries/4096",
            "value": 163353,
            "range": "± 576",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/100",
            "value": 293416,
            "range": "± 3674",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/500",
            "value": 8418086,
            "range": "± 19958",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana index build — DiskANN-style graph construction (dim=32)/nodes/1000",
            "value": 33560744,
            "range": "± 84138",
            "unit": "ns/iter"
          },
          {
            "name": "Vamana query — greedy beam search over 1K-node graph (dim=32, top-10)",
            "value": 14315,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/100",
            "value": 29259650,
            "range": "± 1096610",
            "unit": "ns/iter"
          },
          {
            "name": "WAL append — fsync'd causal edge writes/edges/1000",
            "value": 298843284,
            "range": "± 9978863",
            "unit": "ns/iter"
          },
          {
            "name": "WAL replay — crash recovery: read 1K edges from disk",
            "value": 3027813,
            "range": "± 69346",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_write — single cell persist with fsync (dim=64)",
            "value": 335884,
            "range": "± 79621",
            "unit": "ns/iter"
          },
          {
            "name": "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
            "value": 115009,
            "range": "± 710",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}