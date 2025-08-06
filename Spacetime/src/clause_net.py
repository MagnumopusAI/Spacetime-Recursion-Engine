"""Neural clause satisfaction inspired by SMUG logic.

This module implements :class:`ClauseNet`, a simple network that treats each
SAT clause as a resonant circuit.  Clause activations behave like voltages that
rise when variables align, and global satisfaction mirrors a conservation law
similar to the Preservation Constraint Equation (PCE).  An auxiliary helper,
:func:`build_llm_prompt`, converts low-satisfaction clauses into human readable
hints for corrective action.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import unitary_group


class ClauseNet(nn.Module):
    """Map boolean assignments to clause satisfaction amplitudes.

    Parameters
    ----------
    num_vars:
        Number of boolean variables.  Analogous to the number of terminals in an
        electrical network.
    num_clauses:
        Number of clauses to evaluate.
    use_attention:
        If ``True``, a single-head self-attention layer entangles variables much
        like a control circuit that cross-couples signals.
    random_state:
        Optional seed or generator for deterministic unitary initialization.

    Notes
    -----
    The weight matrix is sampled from the real component of a unitary matrix.
    This preserves the spectral norm, echoing how the PCE enforces invariant
    balance in physical systems.
    """

    def __init__(
        self,
        num_vars: int,
        num_clauses: int,
        use_attention: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.use_attention = use_attention

        U = unitary_group.rvs(num_vars, random_state=random_state)
        self.weights = nn.Parameter(torch.tensor(U.real[:num_clauses], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_clauses))

        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_vars, num_heads=1, batch_first=True
            )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clause-wise and global satisfaction scores.

        Parameters
        ----------
        X:
            Tensor of shape ``(batch, num_vars)`` containing binary assignments.
            Each row is like a particular wiring of switches in an electrical
            board.

        Returns
        -------
        satisfaction, global_satisfaction:
            - ``satisfaction``: tensor ``(batch, num_clauses)`` with values in
              ``[0, 1]`` representing how strongly each clause fires.
            - ``global_satisfaction``: tensor ``(batch,)`` with the product of
              clause scores, analogous to evaluating whether the entire circuit
              remains in balance.
        """

        if self.use_attention:
            X_attn, _ = self.attention(X.unsqueeze(1), X.unsqueeze(1), X.unsqueeze(1))
            X = X_attn.squeeze(1)

        logits = X @ self.weights.T + self.bias
        satisfaction = torch.sigmoid(logits)
        global_satisfaction = satisfaction.prod(dim=1)
        return satisfaction, global_satisfaction


def build_llm_prompt(
    clause_scores: torch.Tensor,
    clause_texts: list[str],
    input_assignment: torch.Tensor,
    threshold: float = 0.9,
) -> str:
    """Translate low-satisfaction clauses into a text prompt.

    Parameters
    ----------
    clause_scores:
        Tensor of clause satisfaction scores ``(num_clauses,)`` for a single
        assignment.
    clause_texts:
        Textual representation for each clause, used for human guidance.
    input_assignment:
        Variable assignment whose gradient guides potential flips.
    threshold:
        Minimum acceptable satisfaction.  Scores below this level are reported.

    Returns
    -------
    str
        Human-readable message summarizing violated clauses and suggesting
        variable flips based on gradient magnitude.  The gradient analysis acts
        like probing which switches most influence the circuit's overall
        balance.
    """

    violated = [
        f"Clause {i + 1}: {clause_texts[i]} (score={score:.2f})"
        for i, score in enumerate(clause_scores.tolist())
        if score < threshold
    ]

    if not violated:
        return "All clauses are satisfied. The lattice hums with harmonic truth."

    prompt = "Violations Detected:\n" + "\n".join(violated)
    prompt += "\n\nTask: Suggest a variable assignment change to restore satisfiability."

    grads = torch.autograd.grad(clause_scores.sum(), input_assignment, retain_graph=True)[0]
    top_vars = torch.topk(torch.abs(grads), k=min(3, grads.numel())).indices.tolist()
    prompt += f"\n\nGradient Hints: Flip variables {[f'x{i + 1}' for i in top_vars]} to perturb the clause landscape."
    return prompt

