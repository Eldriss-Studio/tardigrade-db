# LongMemEval-S Generator

Synthetic dataset generator for the LongMemEval benchmark (simplified).

## What it generates

LongMemEval-S produces 6 task types that test memory capabilities:

| Task | Description | Metric |
|------|-------------|--------|
| `single-session-user` | User reveals facts across turns in one session | Recall |
| `single-session-assistant` | Assistant fact recall from user conversation | Recall |
| `multi-session` | Same user across N separate sessions | Multi-session recall |
| `temporal-reasoning` | Time-ordered events requiring date math | Temporal accuracy |
| `knowledge-update` | Old fact vs new fact (system must prefer updated) | Update correctness |
| `abstention` | Unanswerable question (system should abstain) | Precision/abstention rate |

## Usage

```bash
# Generate all task types (default seed=42)
PYTHONPATH=python python3 -m tdb_bench.datasets.generators.longmemeval_s \
  --output data/longmemeval_s.jsonl

# Generate specific tasks
python3 -m tdb_bench.datasets.generators.longmemeval_s \
  --output data/longmemeval_s_temporal.jsonl \
  --tasks single-session-user,temporal-reasoning

# Generate N items
python3 -m tdb_bench.datasets.generators.longmemeval_s \
  --output data/longmemeval_s_100.jsonl \
  --num-items 100 \
  --seed 123

# Import as library
from tdb_bench.datasets.generators.longmemeval_s import LongMemEvalSDatasetGenerator
gen = LongMemEvalSDatasetGenerator(seed=42)
items = gen.generate(num_items=100, tasks=["temporal-reasoning", "knowledge-update"])
gen.to_jsonl(items, "data/my_tasks.jsonl")
```

## Integration with benchmark runner

The generated JSONL is compatible with `LongMemEvalDatasetAdapter`:

```python
from tdb_bench.datasets.longmemeval import LongMemEvalDatasetAdapter
from tdb_bench.runner import BenchmarkRunner

adapter = LongMemEvalDatasetAdapter(
    revision="synthetic-v1",
    path="data/longmemeval_s_sample.jsonl"
)
runner = BenchmarkRunner.from_config_file(Path("config/benchmark.yaml"))
runner.run(mode="dataset", adapter=adapter, ...)
```

## Task design

### Abstention metric

For `abstention` task, ground_truth is `UNANSWERABLE`. Score is:

- **Correct abstention**: system returns "I don't know" or similar → score 1.0
- **Wrong answer**: system guesses → score 0.0
- **Wrong abstention**: question IS answerable but system abstains → score 0.0

### Knowledge update metric

The `knowledge-update` task requires the system to prefer the **most recent** fact
(old_city: Portland → current_city: Austin). A system that remembers "Portland"
but not "Austin" scores 0.

### Temporal reasoning metric

Questions require date arithmetic. Example:
- First event: 2022-03-15
- Third event: 2024-01-10
- Question: "Days between first and third?" → 666

## Validation

Run against TDB with:

```bash
PYTHONPATH=python python3 -m tdb_bench.run \
  --dataset longmemeval_s \
  --system tardigrade \
  --output results/longmemeval_s_results.json
```