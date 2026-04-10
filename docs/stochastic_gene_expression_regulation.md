# Stochastic Gene Expression Regulation: Markovian Dynamics, Probability Distributions, and Noise-Induced Switching

## 1. Introduction

Gene expression in single cells is inherently stochastic because key molecular species (promoter states, mRNA, proteins) often exist in low copy numbers. In this mesoscopic regime, deterministic mass-action ODEs are insufficient to explain observed cell-to-cell variability. A natural framework is the continuous-time Markov chain (CTMC), where discrete molecular events are random and memoryless.

This stochasticity can be functional, not merely detrimental: it enables phenotypic diversification, bet-hedging, and noise-induced switching in fluctuating environments.

## 2. CTMC Telegraph Model

Consider a two-state promoter model with promoter state \(s\in\{\mathrm{OFF},\mathrm{ON}\}\), mRNA count \(m\), and protein count \(p\). The core reactions are:

- **Promoter activation:** \(\mathrm{OFF}\to\mathrm{ON}\) at rate \(k_{on}\)
- **Promoter deactivation:** \(\mathrm{ON}\to\mathrm{OFF}\) at rate \(k_{off}\)
- **Transcription (ON only):** \(m\to m+1\) at rate \(\alpha\)
- **mRNA degradation:** \(m\to m-1\) at rate \(\gamma_m m\)
- **Translation:** \(p\to p+1\) at rate \(\beta m\)
- **Protein degradation/dilution:** \(p\to p-1\) at rate \(\gamma_p p\)

The joint law \(P_s(m,p,t)\) satisfies coupled Chemical Master Equations (CMEs), one for each promoter state.

## 3. Generating-Function View and Marginal Laws

At stationarity, conditioned on promoter state:

- If promoter is **ON**, mRNA follows Poisson with mean \(\alpha/\gamma_m\)
- If promoter is **OFF**, mRNA collapses toward zero
- Conditioned on \(m\), proteins follow Poisson with mean \((\beta/\gamma_p)m\)

Using probability generating functions (PGFs), the unconditional protein distribution becomes a **compound Poisson** form with over-dispersion. A standard signature is Fano factor \(\mathrm{Var}(p)/\mathbb{E}[p] > 1\).

In the limit \(\gamma_m \gg \gamma_p\), the model approaches a **negative binomial (NB)** law, which explains why NB fits are often successful for single-cell count data.

## 4. Interpreting NB Fits in scRNA-seq

NB agreement alone is not proof of intrinsic transcriptional bursting. Similar marginals can emerge from extrinsic heterogeneity (e.g., cell-to-cell rate variability) and technical sampling/capture noise. Thus, mechanistic interpretation requires either time-resolved data, perturbations, or stronger model selection than marginal steady-state fitting alone.

## 5. Metastability and Noise-Induced Switching

When promoter switching is slow compared with mRNA/protein relaxation (non-adiabatic regime), the distribution can become bimodal, corresponding to low and high expression basins. Switching between these basins is a rare event.

Large-deviation and first-passage methods quantify these transitions. In diffusion-like reductions, Kramers-style behavior predicts exponentially sensitive switching times, typically of form:

\[
\tau_{switch} \sim A\exp(\Delta U / D)
\]

where \(\Delta U\) is an effective barrier and \(D\) is effective noise intensity.

## 6. Numerical Methods for CME and First-Passage Quantities

- **Gillespie SSA:** exact trajectories, expensive for rare events and stiff systems
- **Tau-leaping / stochastic QSSA:** efficient for separated timescales, but can misestimate fast-species noise
- **Finite State Projection (FSP):** truncated CME with controlled error bounds
- **ACME-style truncation/aggregation:** efficient steady-state landscapes for complex multiscale networks
- **Finite-volume / backward-Kolmogorov approaches:** practical route to MFPT and exit-time statistics

## 7. Synthetic Biology Design Implications

Stochastic modeling directly informs circuit engineering:

- Tune burst size/frequency via promoter and translation control
- Use operons to correlate expression and buffer relative noise
- Use CRISPRi/CRISPRa for reversible, programmable repression/activation
- Counter resource competition (winner-take-all) with negative feedback controllers
- Reduce CRISPR leakiness with antisense sequestration motifs

## 8. Conclusion

CTMC/CME models provide a rigorous basis for linking molecular events to whole-cell variability. PGFs clarify why compound Poisson and NB laws are prevalent, while large-deviation and first-passage tools explain rare switching dynamics. Together, these tools support both mechanistic inference and the rational design of robust synthetic circuits.

## Selected references

1. Noise in Biology (NIH/PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC4033672/
2. Stochastic switching in biology (Bressloff): http://www.math.utah.edu/~bresslof/publications/17-1.pdf
3. Analytical distributions for stochastic gene expression (PNAS): https://www.pnas.org/doi/10.1073/pnas.0803850105
4. Evaluating negative binomial models in scRNA-seq (PLOS Comp Biol): https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014014
5. DNA-binding kinetics and noise-induced switching (NIH/PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC4624158/
6. Accurate Chemical Master Equation solution using multi-finite buffers (ACME): https://pmc.ncbi.nlm.nih.gov/articles/PMC5066912/
7. CRISPR-based gene expression control for synthetic circuits: https://pmc.ncbi.nlm.nih.gov/articles/PMC7609024/
