"""Domain-specific language (DSL) helpers for CPQ preservation directives.

The symbolic language mirrors the pre-flight checklist of a spacecraft. Each
command line encodes the parameters that must balance the Cognitive
Preservation Quadratic (CPQ) before a memory pattern can be launched into the
agent's long-term store. By centralizing the syntax rules here we keep the rest
of the system focused on physics-inspired reasoning while this module acts as
the mission control console translating intent into calibrated assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict

from .gate_core import SymbolicInput


PRESERVE_DIRECTIVE = re.compile(r"^\s*preserve\s*\((?P<body>.*)\)\s*$")


class DirectiveSyntaxError(ValueError):
    """Raised when a preservation directive fails syntactic validation."""


@dataclass(frozen=True)
class PreservationDirective:
    """Concrete representation of a symbolic preservation command.

    Attributes
    ----------
    assignments:
        Mapping of the CPQ parameters ``M``, ``alpha``, ``beta``, and ``chi`` to
        the textual expressions provided by the caller. Keeping the original
        expressions intact is analogous to storing raw sensor readings before
        translating them into a shared reference frame.
    """

    assignments: Dict[str, SymbolicInput]

    REQUIRED_KEYS = ("M", "alpha", "beta", "chi")

    def to_assignments(self) -> Dict[str, SymbolicInput]:
        """Return a dictionary compatible with :mod:`gate_core` builders."""

        return dict(self.assignments)


def _split_arguments(body: str) -> Dict[str, str]:
    """Return a mapping extracted from the comma-separated argument payload.

    The splitting routine is intentionally strict so that every command resembles
    a carefully balanced chemical equation. Each term must contain a key-value
    pair separated by ``=`` and we forbid duplicate keys to prevent silent drift
    in the preservation ledger.
    """

    if not body.strip():
        raise DirectiveSyntaxError("Preservation directive requires assignments inside parentheses.")

    entries: Dict[str, str] = {}
    for segment in body.split(','):
        key_value = segment.strip()
        if not key_value:
            continue
        if '=' not in key_value:
            raise DirectiveSyntaxError(f"Missing '=' in segment: {key_value!r}")
        key, value = key_value.split('=', maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise DirectiveSyntaxError("Empty parameter name detected in directive.")
        if not value:
            raise DirectiveSyntaxError(f"Missing value for parameter {key!r}.")
        if key in entries:
            raise DirectiveSyntaxError(f"Duplicate assignment for parameter {key!r}.")
        entries[key] = value
    return entries


def parse_preservation_directive(command: str) -> PreservationDirective:
    """Parse a ``preserve(...)`` command into a :class:`PreservationDirective`.

    The parser treats the directive as an orbital insertion burn: every parameter
    must be present so that the symbolic trajectory hits the preservation
    manifold. Failing to include a required component immediately raises a
    :class:`DirectiveSyntaxError`, preventing malformed inputs from slipping past
    the first defense line of the reasoning engine.
    """

    match = PRESERVE_DIRECTIVE.match(command)
    if not match:
        raise DirectiveSyntaxError("Directive must start with 'preserve(' and end with ')'.")

    entries = _split_arguments(match.group("body"))

    missing = [key for key in PreservationDirective.REQUIRED_KEYS if key not in entries]
    if missing:
        raise DirectiveSyntaxError(f"Directive missing parameters: {', '.join(missing)}")

    assignments: Dict[str, SymbolicInput] = {key: entries[key] for key in PreservationDirective.REQUIRED_KEYS}
    return PreservationDirective(assignments=assignments)

