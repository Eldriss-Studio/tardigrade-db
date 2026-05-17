"""``tardigrade init`` — scaffold a new consumer project.

Lays down a starter file, README, and an empty ``engine_dir/``
into the target directory. Refuses to clobber existing non-empty
directories unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tardigrade_cli.constants import (
    DEFAULT_TEMPLATE,
    ENGINE_DIR_NAME,
    TEMPLATES,
)
from tardigrade_cli.subcommand import Subcommand


_TEMPLATES_ROOT = Path(__file__).parent / "templates"


class InitCommand(Subcommand):
    """Scaffold a new TardigradeDB consumer project."""

    name = "init"
    help = "Scaffold a new TardigradeDB consumer project."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Scaffold a new project directory with a starter file, README, "
            "and an empty engine state directory.\n\n"
            "Example:\n"
            "  tardigrade init --dir ./my-project\n"
            "  tardigrade init --dir ./my-rust --template rust-basic"
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument(
            "--dir",
            required=True,
            help="Target directory for the scaffold (created if missing).",
        )
        parser.add_argument(
            "--template",
            choices=sorted(TEMPLATES.keys()),
            default=DEFAULT_TEMPLATE,
            help=f"Starter template (default: {DEFAULT_TEMPLATE}).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite an existing non-empty directory.",
        )

    def run(self, args: argparse.Namespace) -> int:
        target = Path(args.dir)
        template_dir = _TEMPLATES_ROOT / TEMPLATES[args.template]
        if not template_dir.is_dir():  # pragma: no cover — packaging guard
            print(
                f"error: template '{args.template}' missing at {template_dir}",
                file=sys.stderr,
            )
            return 2

        if target.exists() and any(target.iterdir()) and not args.force:
            print(
                f"error: '{target}' is not empty. Pass --force to overwrite.",
                file=sys.stderr,
            )
            return 1

        target.mkdir(parents=True, exist_ok=True)

        for src in template_dir.rglob("*"):
            rel = src.relative_to(template_dir)
            dest = target / rel
            if src.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

        (target / ENGINE_DIR_NAME).mkdir(exist_ok=True)

        print(f"scaffolded {args.template} at {target}")
        print(f"  → cd {target}")
        if args.template == "python-basic":
            print("  → python main.py")
        elif args.template == "rust-basic":
            print("  → cargo run")
        return 0
