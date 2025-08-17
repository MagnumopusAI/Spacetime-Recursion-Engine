import math
from Spacetime.src.helical_bundle import ideal_alpha_helix, assemble_four_helix_bundle


def test_ideal_alpha_helix_invariants():
    seq = "AAAAAA"
    helix = ideal_alpha_helix(seq)
    radius = 2.3
    rise = 1.5

    # Check radial distance invariant
    for _, _, _, (x, y, _) in helix:
        r = math.hypot(x, y)
        assert math.isclose(r, radius, rel_tol=1e-6)

    # Check axial rise between consecutive residues
    for (_, i1, _, (_, _, z1)), (_, i2, _, (_, _, z2)) in zip(helix, helix[1:]):
        assert i2 - i1 == 1
        assert math.isclose(z2 - z1, rise, rel_tol=1e-6)


def test_assemble_four_helix_bundle_count():
    bundle = [
        ("A", "MKQLEDKVEELLSKNYHLENEVARLKKLV"),
        ("B", "LQALEQKLAQAEKQLLAQAEKQLLAQAE"),
        ("C", "LSQLEAEIQALEQENQALEKEIQALEQELQAVEQE"),
        ("D", "LKALKQKIQALKQKNQAIKQKLQALKQKLQAVKQK"),
    ]
    atoms = assemble_four_helix_bundle(bundle)
    assert len(atoms) == sum(len(seq) for _, seq in bundle)
