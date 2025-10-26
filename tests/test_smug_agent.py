import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.preservation import evaluate_preservation_constraint, solve_tau_from_sigma
from src.smug_stability_agent import SMUGAgent


def test_smug_agent_coherent_action_respects_pce():
    agent = SMUGAgent(M=4, initial_chi_factor=1.5, rng_seed=0, initial_memory_state=np.ones(4))
    observation = np.full(4, 0.5)

    action = agent.compute_action(observation)

    assert action is not None
    last_event = agent.action_log[-1]
    sigma = float(np.linalg.norm(agent.memory_state))
    tau = solve_tau_from_sigma(sigma)
    assert abs(evaluate_preservation_constraint(sigma, tau)) <= 1e-6
    assert last_event["status"] == "COHERENT"
    assert "pce_value" in last_event


def test_smug_agent_collapse_when_chi_subcritical():
    agent = SMUGAgent(M=4, initial_chi_factor=1.5, rng_seed=1, initial_memory_state=np.ones(4))
    agent.chi = agent.chi_crit - 0.5

    observation = np.zeros(4)
    action = agent.compute_action(observation)

    assert action is None
    last_event = agent.action_log[-1]
    assert last_event["status"] == "COLLAPSED"
    assert last_event["discriminant"] < 0


def test_update_memory_preserves_unit_norm():
    agent = SMUGAgent(M=4, initial_chi_factor=1.2, rng_seed=2, initial_memory_state=np.arange(1, 5))
    observation = np.full(4, 0.25)

    agent.update_memory(observation, reward=0.5)

    norm = np.linalg.norm(agent.memory_state)
    assert np.isclose(norm, 1.0)
