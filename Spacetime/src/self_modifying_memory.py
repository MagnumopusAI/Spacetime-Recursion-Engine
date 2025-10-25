"""Self-modifying neural recursion utilities.

This module extends the SMUG recursion framework with a neural component that
respects the Preservation Constraint Equation (PCE).  The learning dynamics are
stabilized by comparing the memory stiffness with the discriminant derived from
an analogue of the PCE, ensuring cognitive states stay within the physical
branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .preservation import check_preservation


def calculate_cognitive_discriminant(memory_stiffness: float) -> float:
    """Return the stability discriminant for a given memory stiffness.

    The discriminant ``2 * sqrt(M)`` mirrors evaluating the safety margin of a
    suspension bridge: if the learning coupling exceeds this margin the
    structure remains stable, otherwise oscillations may grow without bound.
    """

    if memory_stiffness <= 0:
        raise ValueError("Memory stiffness must remain positive.")
    return 2.0 * sqrt(memory_stiffness)


def evaluate_preservation_alignment(memory_stiffness: float) -> float:
    """Measure how the memory stiffness aligns with the PCE solution.

    The returned ``tau`` represents how much torsion the memory lattice can
    sustain.  In everyday terms, it is akin to checking the rated load for a
    crane before hoisting a heavy object.
    """

    tau, _ = check_preservation(memory_stiffness)
    return tau


def derive_learning_rate(coupling: float, preservation_tau: float) -> float:
    """Compute a learning rate honoring the PCE alignment.

    Averaging ``coupling`` with ``preservation_tau`` is akin to blending hot and
    cold water to obtain a comfortable bath temperature: both sources must flow
    for the mixture to be usable.
    """

    if coupling <= 0 or preservation_tau <= 0:
        raise ValueError("Coupling and preservation tau must remain positive.")
    return 0.001 * (coupling + preservation_tau) / 2.0


@dataclass
class MemoryInvariant:
    """Container storing the physical invariants of the learning lattice.

    Attributes
    ----------
    stiffness:
        Analogue of ``sigma`` in the PCE capturing how rigid the long-term
        memory should remain.
    coupling:
        Learning coupling analogous to ``tau`` that determines how strongly new
        observations reshape the network.
    discriminant:
        Critical threshold ensuring the self-modification loop stays within the
        physical branch of the recursion.
    preservation_tau:
        Reference torsion returned by :func:`evaluate_preservation_alignment`.
    """

    stiffness: float = 1.0
    coupling: float = 3.0
    discriminant: float = field(init=False)
    preservation_tau: float = field(init=False)

    def __post_init__(self) -> None:
        if self.stiffness <= 0:
            raise ValueError("Memory stiffness must remain positive to satisfy the PCE analogue.")
        self.discriminant = calculate_cognitive_discriminant(self.stiffness)
        self.preservation_tau = evaluate_preservation_alignment(self.stiffness)


class SelfModifyingNN(nn.Module):
    """Self-modifying neural network with persistent contextual memory.

    The module behaves like a mechanical gyroscope: the network's weights form
    the rotor, while the memory invariant acts as a gimbal that keeps the rotor
    from wobbling into unstable trajectories.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        memory_invariant: Optional[MemoryInvariant] = None,
    ) -> None:
        super().__init__()
        self.base_input_size = input_size
        self.output_size = output_size
        self.memory_invariant = memory_invariant or MemoryInvariant()
        augmented_size = self.base_input_size + self.output_size

        self.fc1 = nn.Linear(augmented_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.memory: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, context: Optional[str] = None) -> torch.Tensor:
        """Forward pass respecting contextual memory augmentation.

        Parameters
        ----------
        x:
            Input tensor shaped ``(batch, input_size)`` analogous to sensory
            observations.
        context:
            Key describing the environmental setting.
        """

        x = self._ensure_batch_shape(x)
        memory_state = self._assemble_memory_state(x, context)
        augmented_input = torch.cat((x, memory_state), dim=-1)
        hidden = torch.relu(self.fc1(augmented_input))
        return self.fc2(hidden)

    def update_weights(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        context: Optional[str] = None,
    ) -> None:
        """Update the network weights using a stability-aware Adam step.

        The process mirrors tuning a radio: the discriminant bounds prevent us
        from over-rotating the dial, while the Adam optimizer zeroes in on the
        correct signal.
        """

        threshold = self.memory_invariant.discriminant
        if self.memory_invariant.coupling < threshold:
            raise ValueError(
                "Learning coupling below critical threshold; risk of cognitive collapse."
            )

        learning_rate = derive_learning_rate(
            self.memory_invariant.coupling, self.memory_invariant.preservation_tau
        )
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        output = self.forward(input_data, context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if context is not None:
            self.memory[context] = output.detach().clone()

    def save_memory(self, filepath: str) -> None:
        """Persist contextual memory states to disk.

        Similar to writing a lab notebook, this call stores the latest
        observations for later experiments.
        """

        torch.save(self.memory, filepath)

    def load_memory(self, filepath: str) -> None:
        """Load contextual memory states from disk.

        This is analogous to rereading earlier lab notes before continuing a
        sequence of experiments.
        """

        self.memory = torch.load(filepath)

    def _ensure_batch_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Guarantee that inputs are two-dimensional.

        Treats a single observation like a lone marble placed into an egg carton:
        the marble gains the extra axis so it sits neatly among other samples.
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _assemble_memory_state(
        self, x: torch.Tensor, context: Optional[str]
    ) -> torch.Tensor:
        """Construct the memory tensor associated with a context.

        Reuses stored outputs if available; otherwise returns zeros, just like a
        chalkboard that either preserves previous scribbles or is wiped clean.
        """

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if context is None or context not in self.memory:
            return torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)

        memory_state = self.memory[context]
        memory_state = self._ensure_batch_shape(memory_state)
        if memory_state.shape[0] != batch_size:
            memory_state = memory_state.expand(batch_size, -1)
        return memory_state.to(device=device, dtype=dtype)
