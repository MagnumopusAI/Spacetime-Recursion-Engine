"""Atomic data structures for toy chemical modeling.

This module introduces simple dataclasses for representing atoms and
collections of atoms. The goal is to provide a minimal yet physically
inspired container that mirrors how chemists build molecules from
individual atoms.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Atom:
    """Basic atom with a name and mass.

    Parameters
    ----------
    name:
        Chemical symbol, e.g. ``"H"`` for hydrogen.
    mass:
        Atomic mass in unified atomic mass units.
    """

    name: str
    mass: float


@dataclass
class AtomCollectionDataclasses:
    """Container for a set of :class:`Atom` objects.

    The collection emulates a beaker where atoms are dropped one by one
    before forming more complex structures. It intentionally keeps the
    interface minimal while demonstrating how dataclasses can manage
    mutable collections cleanly.
    """

    atoms: List[Atom] = field(default_factory=list)

    def add_atom(self, atom: Atom) -> None:
        """Add an atom to the collection."""
        self.atoms.append(atom)

