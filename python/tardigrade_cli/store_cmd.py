"""``tardigrade store`` — store a single fact via TardigradeClient."""

from __future__ import annotations

import argparse

from tardigrade_cli.constants import DEFAULT_OWNER
from tardigrade_cli.subcommand import Subcommand


class StoreCommand(Subcommand):
    """Store a single fact in the engine."""

    name = "store"
    help = "Store a single fact in the engine."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Store one fact in the engine at --dir under the given owner.\n\n"
            "Example:\n"
            "  tardigrade store --dir ./engine \"Alice moved to Berlin in 2021.\""
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument("--dir", required=True, help="Engine state directory.")
        parser.add_argument(
            "--owner",
            type=int,
            default=DEFAULT_OWNER,
            help=f"Owner id (default: {DEFAULT_OWNER}).",
        )
        parser.add_argument("text", help="The fact text to store.")

    def run(self, args: argparse.Namespace) -> int:
        from tardigrade_hooks import TardigradeClient

        client = TardigradeClient(db_path=args.dir, owner=args.owner)
        pack_id = client.store(args.text)
        print(f"stored pack #{pack_id} (owner={args.owner})")
        return 0
