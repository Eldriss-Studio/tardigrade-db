# TardigradeDB starter (Python)

A minimal example showing how to store and retrieve a memory.

## Run

```bash
pip install tardigrade-db
python main.py
```

## Layout

- `main.py` — the example program. Opens an engine in `./engine_dir/`,
  stores one fact, queries it back.
- `engine_dir/` — persistent storage. Safe to delete to start fresh.

## Next steps

- Read [`docs/guide/consumers.md`](https://github.com/Eldriss-Studio/tardigrade-db/blob/main/docs/guide/consumers.md)
  for the full consumer guide.
- Try `tardigrade query "your query here"` from this directory.
- See `tardigrade --help` for all CLI subcommands.
