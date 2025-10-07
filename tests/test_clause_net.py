import numpy as np
import pytest
import sys
from pathlib import Path

try:  # pragma: no cover - skip if torch unavailable
    import torch
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("torch is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.clause_net import ClauseNet, build_llm_prompt


def test_clause_net_global_product():
    np.random.seed(0)
    torch.manual_seed(0)
    model = ClauseNet(num_vars=4, num_clauses=3, random_state=0)
    X = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32
    )
    satisfaction, global_satisfaction = model(X)
    expected = satisfaction.prod(dim=1)
    assert torch.allclose(global_satisfaction, expected)


def test_build_llm_prompt_violation():
    np.random.seed(0)
    torch.manual_seed(0)
    model = ClauseNet(num_vars=4, num_clauses=3, random_state=0)
    clause_texts = ["(x1 ∨ x2)", "(¬x3 ∨ x4)", "(x1 ∨ ¬x2)"]
    X = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)
    clause_scores, _ = model(X.unsqueeze(0))
    prompt = build_llm_prompt(clause_scores[0], clause_texts, X, threshold=0.95)
    assert "Violations Detected" in prompt
    assert "Gradient Hints" in prompt


def test_build_llm_prompt_all_satisfied():
    np.random.seed(0)
    torch.manual_seed(0)
    model = ClauseNet(num_vars=4, num_clauses=3, random_state=0)
    clause_texts = ["(x1 ∨ x2)", "(¬x3 ∨ x4)", "(x1 ∨ ¬x2)"]
    X = torch.tensor([1.0, 1.0, 0.0, 1.0], requires_grad=True)
    clause_scores, _ = model(X.unsqueeze(0))
    prompt = build_llm_prompt(clause_scores[0], clause_texts, X, threshold=0.0)
    assert prompt.startswith("All clauses are satisfied")
