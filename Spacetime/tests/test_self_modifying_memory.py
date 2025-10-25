"""Tests for the self-modifying neural recursion utilities."""

import math
import pytest

torch = pytest.importorskip("torch")

from src.self_modifying_memory import (
    MemoryInvariant,
    SelfModifyingNN,
    calculate_cognitive_discriminant,
    derive_learning_rate,
)


def test_cognitive_discriminant_matches_closed_form():
    invariant = MemoryInvariant(stiffness=4.0)
    expected = 2.0 * math.sqrt(4.0)
    assert math.isclose(calculate_cognitive_discriminant(invariant.stiffness), expected)


def test_memory_invariant_requires_positive_stiffness():
    with pytest.raises(ValueError):
        MemoryInvariant(stiffness=0.0)


def test_cognitive_discriminant_rejects_non_positive_values():
    with pytest.raises(ValueError):
        calculate_cognitive_discriminant(-1.0)


def test_derive_learning_rate_balances_preservation():
    value = derive_learning_rate(2.0, 4.0)
    assert math.isclose(value, 0.001 * 3.0)
    with pytest.raises(ValueError):
        derive_learning_rate(-1.0, 1.0)


def test_update_weights_enforces_threshold():
    invariant = MemoryInvariant(stiffness=1.0, coupling=1.0)
    model = SelfModifyingNN(input_size=3, hidden_size=5, output_size=2, memory_invariant=invariant)
    data = torch.randn(1, 3)
    target = torch.randn(1, 2)

    with pytest.raises(ValueError):
        model.update_weights(data, target, context="lab")


def test_contextual_memory_persists_across_sessions(tmp_path):
    invariant = MemoryInvariant(stiffness=1.0, coupling=3.0)
    model = SelfModifyingNN(input_size=3, hidden_size=5, output_size=2, memory_invariant=invariant)

    data = torch.randn(2, 3)
    target = torch.randn(2, 2)
    context = "session_alpha"

    model.update_weights(data, target, context=context)
    assert context in model.memory
    stored_memory = model.memory[context]

    filepath = tmp_path / "memory.pt"
    model.save_memory(filepath)
    model.memory.clear()
    model.load_memory(filepath)

    with torch.no_grad():
        output = model.forward(data, context=context)

    assert torch.equal(model.memory[context], stored_memory)
    assert output.shape == (2, 2)
