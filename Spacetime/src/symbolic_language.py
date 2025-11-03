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
from typing import Dict, List, Optional

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

    The parser behaves like a spectrometer aligning multiple diffraction gratings:
    it tracks nested parentheses, brackets, and quoted strings so that commas
    buried inside function calls or matrices do not trigger premature splits.
    This keeps rich symbolic expressions intact while still enforcing the strict
    bookkeeping required by the preservation directive.
    """

    if not body.strip():
        raise DirectiveSyntaxError("Preservation directive requires assignments inside parentheses.")

    openings = {'(': ')', '[': ']', '{': '}'}
    stack: List[str] = []
    entries: Dict[str, str] = {}
    buffer: List[str] = []
    quote: Optional[str] = None

    def flush_segment() -> None:
        segment = ''.join(buffer).strip()
        buffer.clear()
        if not segment:
            return
        if '=' not in segment:
            raise DirectiveSyntaxError(f"Missing '=' in segment: {segment!r}")
        key, value = segment.split('=', maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise DirectiveSyntaxError("Empty parameter name detected in directive.")
        if not value:
            raise DirectiveSyntaxError(f"Missing value for parameter {key!r}.")
        if key in entries:
            raise DirectiveSyntaxError(f"Duplicate assignment for parameter {key!r}.")
        entries[key] = value

    for char in body:
        if quote:
            buffer.append(char)
            if char == quote and (len(buffer) < 2 or buffer[-2] != '\\'):
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
            continue

        if char in openings:
            stack.append(openings[char])
            buffer.append(char)
            continue

        if stack and char == stack[-1]:
            stack.pop()
            buffer.append(char)
            continue

        if char == ',' and not stack:
            flush_segment()
            continue

        buffer.append(char)

    if quote:
        raise DirectiveSyntaxError("Unterminated string literal in directive body.")
    if stack:
        raise DirectiveSyntaxError("Unbalanced delimiters detected in directive body.")

    flush_segment()
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

