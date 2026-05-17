"""``tardigrade query`` — retrieve top-k matching facts from the engine."""

from __future__ import annotations

import argparse

from tardigrade_cli.constants import DEFAULT_OWNER, DEFAULT_QUERY_TOP_K
from tardigrade_cli.subcommand import Subcommand


class QueryCommand(Subcommand):
    """Retrieve the top-k matches for a query string."""

    name = "query"
    help = "Retrieve the top-k matches for a query."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Query the engine for top-k facts matching the input text.\n\n"
            "Example:\n"
            "  tardigrade query --dir ./engine --top-k 5 \"where did Alice go?\""
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument("--dir", required=True, help="Engine state directory.")
        parser.add_argument(
            "--owner",
            type=int,
            default=DEFAULT_OWNER,
            help=f"Owner id (default: {DEFAULT_OWNER}).",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=DEFAULT_QUERY_TOP_K,
            help=f"Number of results to retrieve (default: {DEFAULT_QUERY_TOP_K}).",
        )
        parser.add_argument("text", help="The query text.")

    def run(self, args: argparse.Namespace) -> int:
        from tardigrade_hooks import TardigradeClient

        client = TardigradeClient(db_path=args.dir, owner=args.owner)
        results = client.query(args.text, k=args.top_k)
        if not results:
            print("no results")
            return 0
        print(f"{len(results)} result(s):")
        for i, hit in enumerate(results, 1):
            score = hit.get("score", 0.0) if isinstance(hit, dict) else 0.0
            text = hit.get("text", "") if isinstance(hit, dict) else str(hit)
            print(f"  [{i}] score={score:.4f}  {text}")
        return 0
