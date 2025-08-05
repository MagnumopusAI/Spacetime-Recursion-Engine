import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.neural_symbolic_compiler import NeuralSymbolicCompiler, ClauseProjector


class MockSMUGArchitect:
    """Provide fixed qubit reliabilities for testing."""

    def get_hardware_calibration_data(self):
        return {
            "q1": 0.99,
            "q2": 0.76,
            "q3": 0.94,
            "q4": 0.89,
            "q5": 0.70,
            "q6": 0.97,
            "q7": 0.88,
            "q8": 0.84,
            "q9": 0.93,
        }


class TestTauAwareCompiler(unittest.TestCase):
    def setUp(self):
        architect = MockSMUGArchitect()
        self.compiler = NeuralSymbolicCompiler(architect)
        self.clause = ClauseProjector(
            "C5_DEMO",
            [(1, False), (2, True), (3, False), (4, True), (5, False)],
        )

    def test_dynamic_threshold(self):
        self.assertAlmostEqual(self.compiler._dynamic_threshold(1.05), 0.905)
        self.assertAlmostEqual(self.compiler._dynamic_threshold(0.81), 0.881)

    def test_compile_high_and_low_tau_fail(self):
        self.assertIsNone(self.compiler.compile(self.clause, 1.05))
        self.assertIsNone(self.compiler.compile(self.clause, 0.81))

    def test_compile_lower_tau_succeeds(self):
        sequence = self.compiler.compile(self.clause, 0.70)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 5)


if __name__ == "__main__":
    unittest.main()
