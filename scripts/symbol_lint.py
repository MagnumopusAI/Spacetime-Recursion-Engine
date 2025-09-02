"""
Ensures LaTeX documents only use macros defined in ``smug_macros.sty``.
Like checking that every structural beam is accounted for before assembling a bridge,
this script preserves symbolic integrity akin to the Preservation Constraint Equation (PCE).
"""

from pathlib import Path
import re
import sys


def load_defined_macros(macros_path: Path) -> set:
    """Catalog macros declared in the shared style file, similar to logging conserved charges before an experiment."""
    text = macros_path.read_text(encoding="utf-8") if macros_path.exists() else ""
    pattern = re.compile(r"\\(?:newcommand|def)\s*(?:\\([A-Za-z]+)|\{\\([A-Za-z]+)\})")
    matches = pattern.findall(text)
    return {m[0] or m[1] for m in matches}


def extract_used_macros(tex_path: Path) -> set:
    """Map every macro used in a LaTeX document, like charting stars before navigating space."""
    pattern = re.compile(r"\\([A-Za-z]+)")
    text = tex_path.read_text(encoding="utf-8")
    return set(pattern.findall(text))


def find_undefined_macros(root: Path, macros_path: Path) -> dict:
    """Return a mapping of files to macros that lack definitions, preserving symbolic invariants across docs."""
    allowed = load_defined_macros(macros_path)
    builtins = {
        "documentclass", "usepackage", "begin", "end", "section", "subsection",
        "label", "ref", "cite", "item", "textbf", "textit", "emph", "title", "author",
        "date", "maketitle", "tableofcontents", "paragraph", "subparagraph",
    }
    undefined = {}
    for tex_file in root.rglob("*.tex"):
        used = extract_used_macros(tex_file)
        local_definitions = load_defined_macros(tex_file)
        missing = sorted(
            macro
            for macro in used
            if macro not in allowed and macro not in builtins and macro not in local_definitions
        )
        if missing:
            undefined[tex_file] = missing
    return undefined


def main() -> None:
    """CLI entry point that exits nonzero if any undefined macros are found."""
    project_root = Path(".")
    macros_file = Path("smug_macros.sty")
    missing = find_undefined_macros(project_root, macros_file)
    if missing:
        for path, macros in missing.items():
            for macro in macros:
                print(f"Undefined macro {macro} in {path}")
        sys.exit(1)
    print("All macros properly defined.")


if __name__ == "__main__":
    main()
