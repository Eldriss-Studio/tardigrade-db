"""Abstract base for CLI subcommands.

Pattern: Strategy + Template Method. Each subcommand is a concrete
``Subcommand`` subclass with ``name``, ``help``, and a ``run`` that
returns an integer exit code. The main argparse dispatcher (in
:mod:`tardigrade_cli.main`) iterates the registered subcommands and
wires them into the parser.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


class Subcommand(ABC):
    """A single CLI subcommand.

    Subclasses must set the class attributes :attr:`name` and
    :attr:`help`, and implement :meth:`add_arguments` and
    :meth:`run`. The exit code returned from :meth:`run` is the
    process exit code (``0`` = success).
    """

    name: str = ""
    help: str = ""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add subcommand-specific flags + positionals to ``parser``."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Execute the subcommand. Return the process exit code."""
