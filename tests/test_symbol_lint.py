"""Tests for the symbol_lint module ensuring symbolic consistency like a conservation law."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.symbol_lint import find_undefined_macros


def test_find_undefined_macros_identifies_missing_macros(tmp_path):
    """An undefined macro is analogous to an unbalanced force and should be detected."""
    macros = tmp_path / "smug_macros.sty"
    macros.write_text("\\newcommand{\\foo}{bar}")
    tex = tmp_path / "doc.tex"
    tex.write_text("\\documentclass{article}\n\\foo\n\\bar")
    missing = find_undefined_macros(tmp_path, macros)
    assert tex in missing and "bar" in missing[tex]


def test_find_undefined_macros_passes_when_all_defined(tmp_path):
    """When every symbol is defined, the system remains in equilibrium."""
    macros = tmp_path / "smug_macros.sty"
    macros.write_text("\\newcommand{\\foo}{bar}")
    tex = tmp_path / "doc.tex"
    tex.write_text("\\documentclass{article}\n\\foo")
    missing = find_undefined_macros(tmp_path, macros)
    assert missing == {}
