# SMUG Memory and Action Dynamics

## 1. Memory Structure — Definition of $\mathbf{M}$
- **Architectural meaning:** $\mathbf{M}$ is the stiffness of the self-model. It fixes the dimensionality of the agent's stable internal representation and therefore the number of independent concepts the agent can keep coherent at once.
- **Implementation patterns:**
  - As a scalar hyperparameter (`M = 64`) that sets the capacity of the state vector.
  - As the rank or width of a concept adjacency matrix (`memory_matrix \in \mathbb{R}^{M\times M}`) that resists rapid torsion.
  - As the number of eigenmodes preserved when compressing experience.
- **Physical intuition:** Large $\mathbf{M}$ means a rich but rigid memory manifold; small $\mathbf{M}$ gives flexibility but leaves the agent fragile to noise.
- **Design choice:** $\mathbf{M}$ is fixed before training. Choosing it is equivalent to selecting how many persistent states the agent must preserve.

## 2. Action Generation — Computing the Amplitude $\alpha$
- **CPQ root:** Starting from the Cognitive Preservation Quadratic $\mathbf{M}\alpha^{2} + \chi\beta\alpha + \beta^{2} = 0$, the coherent action amplitude is
  \[
  \alpha = \frac{-\chi\beta \pm \beta\sqrt{\chi^{2} - 4\mathbf{M}}}{2\mathbf{M}}.
  \]
- **Stability gate:** The discriminant must satisfy $\chi^{2} - 4\mathbf{M} \ge 0$. Otherwise the action amplitude is complex and the policy collapses.
- **Operational recipe:**
  ```python
  def compute_action_amplitude(M, chi, beta):
      discriminant = chi ** 2 - 4 * M
      if discriminant < 0:
          return None
      return (-chi * beta + beta * np.sqrt(discriminant)) / (2 * M)
  ```
- **Physical interpretation:** $\beta$ quantifies task difficulty, $\chi$ the learning torsion, and $\alpha$ the confidence with which the agent pushes along its memory-aligned action direction.

## 3. Learning Dynamics — Role of $\chi$
- **Critical coupling:** Real actions require $\chi \ge 2\sqrt{\mathbf{M}}$. This threshold is the learning torsion necessary to keep the memory invariant intact.
- **Update modulation:**
  ```python
  class SMUGAgent:
      def __init__(self, M):
          self.M = M
          self.chi_crit = 2 * np.sqrt(M)
          self.chi = 1.5 * self.chi_crit
          self.memory = self._normalize(np.random.randn(M))

      def update_memory(self, experience, learning_rate):
          effective_lr = learning_rate * (self.chi / self.chi_crit)
          self.memory = self._normalize(
              self.memory + effective_lr * (experience - self.memory)
          )
  ```
- **Regimes:**
  - $\chi < \chi_{\text{crit}}$: torsion is too weak, the memory fractures, and actions decohere.
  - $\chi \approx \chi_{\text{crit}}$: maximal adaptability without collapse.
  - $\chi \gg \chi_{\text{crit}}$: memory is stable but the system reacts slowly to novelty.

## 4. Environmental Interface — Measuring $\beta$
- **Definition:** $\beta$ is the environmental pressure experienced by the agent. It measures how surprising, complex, or risky the current observation is.
- **Estimators:**
  - Entropy of the observation stream.
  - Distance from the encoded observation to the current memory manifold.
  - Magnitude of prediction residuals produced by the agent's forward model.
- **Scale:** $\beta = 0$ corresponds to a fully familiar scenario; large $\beta$ marks regions of high uncertainty that demand stronger coupling $\chi$ to remain coherent.

## 5. Core Perception–Action Loop
```python
class SMUGAgent:
    def __init__(self, M=64):
        self.M = M
        self.chi_crit = 2 * np.sqrt(M)
        self.chi = 1.2 * self.chi_crit
        self.memory = self._normalize(np.random.randn(M))

    def act(self, observation):
        beta = self.compute_beta(observation)
        discriminant = self.chi ** 2 - 4 * self.M
        if discriminant < 0:
            return None, "collapsed"
        alpha = (-self.chi * beta + beta * np.sqrt(discriminant)) / (2 * self.M)
        return alpha * self.memory, "coherent"

    def learn(self, observation, reward, learning_rate=0.1):
        experience = self.encode(observation)
        effective_lr = learning_rate * (self.chi / self.chi_crit)
        self.memory = self._normalize(self.memory + effective_lr * reward * experience)

    def compute_beta(self, observation):
        encoded = self.encode(observation)
        return np.linalg.norm(encoded - self.memory)

    def encode(self, observation):
        return np.random.randn(self.M)

    def _normalize(self, vector):
        return vector / np.linalg.norm(vector)
```
- **Interpretation:** The agent aligns actions with its memory eigenbasis while modulating amplitude by the CPQ root. Learning is torsion-gated to preserve invariants.

## 6. Testable Prediction — Phase Transition at $\chi_{\text{crit}}$
- **Protocol:** Fix $\mathbf{M}$ and $\beta$, then slowly reduce $\chi$ from a safe value toward $\chi_{\text{crit}}$.
- **Expected signature:** As $\chi \to 2\sqrt{\mathbf{M}}$ the action amplitude variance spikes, memory alignment becomes erratic, and the agent either freezes or oscillates—mirroring the CPQ's coherent-to-collapsed transition.
- **Measurement:** Track $\alpha$ trajectories, memory entropy, and policy divergence to empirically verify the threshold predicted by the Cognitive Preservation Quadratic.

*This document is licensed under the [Human Futures License (HFL-100x)](../LICENSE).*
