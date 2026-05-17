"""``tardigrade consolidate`` — trigger memory consolidation sweep."""

from __future__ import annotations

import argparse

from tardigrade_cli.constants import DEFAULT_OWNER
from tardigrade_cli.subcommand import Subcommand


class ConsolidateCommand(Subcommand):
    """Run a consolidation sweep over the engine."""

    name = "consolidate"
    help = "Trigger memory consolidation and report views attached."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Run a consolidation sweep (multi-view attachment for tier-gated "
            "packs) and report the number of views added.\n\n"
            "Example:\n"
            "  tardigrade consolidate --dir ./engine --owner 1"
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument("--dir", required=True, help="Engine state directory.")
        parser.add_argument(
            "--owner",
            type=int,
            default=DEFAULT_OWNER,
            help=f"Owner id (default: {DEFAULT_OWNER}).",
        )

    def run(self, args: argparse.Namespace) -> int:
        from tardigrade_hooks import TardigradeClient

        client = TardigradeClient(db_path=args.dir, owner=args.owner)
        result = client.consolidate_all()
        # ``consolidate_all`` returns ``dict[int, int]`` keyed by pack
        # id with the number of views attached to that pack.
        total_views = sum(result.values()) if isinstance(result, dict) else 0
        pack_touched = len(result) if isinstance(result, dict) else 0
        print(f"consolidated {pack_touched} pack(s); attached {total_views} view(s)")
        return 0
