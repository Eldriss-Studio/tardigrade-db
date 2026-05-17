"""``tardigrade`` CLI entry point.

Dispatches to registered :class:`Subcommand` instances via argparse.
The list of subcommands is kept short and intentional.
"""

from __future__ import annotations

import argparse
import sys

from tardigrade_cli.chat_cmd import ChatCommand
from tardigrade_cli.consolidate_cmd import ConsolidateCommand
from tardigrade_cli.init_cmd import InitCommand
from tardigrade_cli.query_cmd import QueryCommand
from tardigrade_cli.status_cmd import StatusCommand
from tardigrade_cli.store_cmd import StoreCommand
from tardigrade_cli.subcommand import Subcommand


def _build_parser(subcommands: list[Subcommand]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tardigrade",
        description="TardigradeDB command-line interface.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in subcommands:
        p = sub.add_parser(cmd.name, help=cmd.help)
        cmd.add_arguments(p)
        p.set_defaults(_subcommand=cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the chosen subcommand.

    Returns the subcommand's exit code; ``2`` is reserved for
    argparse errors (which print their own message and `SystemExit`).
    """
    subcommands: list[Subcommand] = [
        InitCommand(),
        StoreCommand(),
        QueryCommand(),
        StatusCommand(),
        ConsolidateCommand(),
        ChatCommand(),
    ]
    parser = _build_parser(subcommands)
    args = parser.parse_args(argv)
    cmd: Subcommand = args._subcommand
    return cmd.run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
