"""``tardigrade status`` — human-readable engine snapshot."""

from __future__ import annotations

import argparse

from tardigrade_cli.constants import DEFAULT_OWNER
from tardigrade_cli.subcommand import Subcommand


class StatusCommand(Subcommand):
    """Print a human-readable summary of the engine's current state."""

    name = "status"
    help = "Show a human-readable engine status snapshot."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Display engine state: cell count, pack count, arena bytes, "
            "tier distribution, owner count.\n\n"
            "Example:\n"
            "  tardigrade status --dir ./engine"
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument("--dir", required=True, help="Engine state directory.")
        parser.add_argument(
            "--owner",
            type=int,
            default=DEFAULT_OWNER,
            help=f"Owner id used to open the client (default: {DEFAULT_OWNER}).",
        )

    def run(self, args: argparse.Namespace) -> int:
        from tardigrade_hooks import TardigradeClient

        client = TardigradeClient(db_path=args.dir, owner=args.owner)
        engine = client.engine

        status = engine.status() if hasattr(engine, "status") else {}
        # `status` returns a dict-like; coerce to dict for stable iteration.
        if hasattr(status, "items"):
            entries = dict(status.items()) if not isinstance(status, dict) else status
        elif hasattr(status, "__dict__"):
            entries = vars(status)
        else:
            entries = {"raw": str(status)}

        pack_count = client.pack_count()

        print(f"engine_dir:    {args.dir}")
        print(f"owner:         {args.owner}")
        print(f"packs (owner): {pack_count}")
        for key in sorted(entries.keys()):
            val = entries[key]
            print(f"  {key}: {val}")
        return 0
