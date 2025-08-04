import sys
import unittest
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.smug_architect import (
    SMUGArchitect,
    generate_3sat,
    auto_verify_3sat,
)


class TestSMUGArchitect(unittest.TestCase):
    def test_generate_3sat_size(self):
        instance = generate_3sat(3, 5)
        self.assertEqual(len(instance), 5)
        for clause in instance:
            self.assertEqual(len(clause), 3)
            for var, val in clause:
                self.assertTrue(1 <= var <= 3)
                self.assertIsInstance(val, bool)

    def test_verify_and_compute(self):
        sat = [
            [(1, True), (2, False), (3, False)],
            [(1, False), (2, False), (3, False)],
            [(1, False), (2, True), (3, True)],
        ]
        arch = SMUGArchitect()
        assignment = arch.compute_artifact("demo", sat, mode="formalist")
        self.assertTrue(arch.verify_assignment(sat, assignment))

    def test_instantiate_computational_manifold(self):
        arch = SMUGArchitect()
        lattice = arch.instantiate_computational_manifold(5)
        self.assertEqual(lattice["volume"], 5 ** 4)
        self.assertEqual(arch.lattice["dim"], 5)

    def test_auto_verify_3sat(self):
        sat = [
            [(1, True), (2, False), (3, False)],
            [(1, False), (2, False), (3, False)],
            [(1, False), (2, True), (3, True)],
        ]
        f_res, s_res = auto_verify_3sat(sat)
        arch = SMUGArchitect()
        self.assertTrue(arch.verify_assignment(sat, f_res))
        self.assertTrue(arch.verify_assignment(sat, s_res))

    def test_design_custom_particle(self):
        arch = SMUGArchitect()
        blueprint = arch.design_custom_particle("Justino", 0.2, 0.5)
        expected_g = math.sqrt(-1.0 / math.log(0.2))
        self.assertIn("Justino", arch.blueprints)
        self.assertAlmostEqual(blueprint["coupling_g"], expected_g)
        with self.assertRaises(ValueError):
            arch.design_custom_particle("Bad", 1.2, 0.5)


if __name__ == "__main__":
    unittest.main()
