# TardigradeDB starter (Rust)

Minimal Rust starter for embedding the TardigradeDB engine.

## Run

```bash
cargo run
```

## Layout

- `Cargo.toml` — depends on `tdb-engine` and `tdb-core`.
- `src/main.rs` — opens the engine at `./engine_dir/`.
- `engine_dir/` — persistent storage. Safe to delete to start fresh.

## Next steps

- Read [`docs/guide/consumers.md`](https://github.com/Eldriss-Studio/tardigrade-db/blob/main/docs/guide/consumers.md).
- The `tdb-engine` crate exposes `Engine::open`, `mem_write_pack`,
  `mem_read_pack`. See the crate docs for the full API.
