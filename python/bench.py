"""Thin wrapper so CI/local can run benchmark CLI with a stable path."""

from tdb_bench.cli import main

raise SystemExit(main())
