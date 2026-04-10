# The PCE Periodic Table: A Complete Derivation from First Principles

## Abstract

This note captures the Preservation Constraint Equation (PCE) framework as provided in the working manuscript and organizes it into a single, non-duplicated reference document.

\[
\mathcal{O}_{\mathrm{res}}(\sigma,\tau,\upsilon) = -2\sigma^2 + 2\tau^2 + 3\tau = \upsilon
\]

The claimed mapping is:

- \(\sigma \to n\) (principal quantum number, with screening corrections via \(Z_{\mathrm{eff}}/n\))
- \(\tau \to \ell\) (orbital angular momentum quantum number)
- \(\upsilon\) as an effective stability tension coordinate

## 1) Orbital-Constraint Layer (Existence)

Using \(\upsilon\le 0\) for bound states:

\[
-2n^2 + 2\ell^2 + 3\ell \le 0
\]

\[
\ell(2\ell+3) \le 2n^2
\]

Hence the existence bound:

\[
\ell_{\max}^{\mathrm{exists}}(n)=\left\lfloor\frac{-3+\sqrt{9+16n^2}}{4}\right\rfloor
\]

This criterion is interpreted as: which \((n,\ell)\) orbitals are mathematically admissible.

## 2) Filling Layer (Ground-State Ordering)

The manuscript introduces the ordering functional:

\[
E(n,\ell)\propto n^2-\ell^2-\frac{3\ell}{2}
\]

with screening-modified form:

\[
E(n,\ell,Z_{\mathrm{eff}})\propto\left(\frac{Z_{\mathrm{eff}}}{n}\right)^2-\ell^2-\frac{3\ell}{2}
\]

and states this reproduces Madelung-style ordering (primary \(n+\ell\), secondary \(n\)).

## 3) Period Capacities

Block capacity for a fixed \(\ell\):

\[
N_{\mathrm{block}}(\ell)=2(2\ell+1)
\]

Claimed period capacities (ground-state periodic table):

- Period 1: 2
- Period 2: 8
- Period 3: 8
- Period 4: 18
- Period 5: 18
- Period 6: 32
- Period 7: 32

## 4) Resolving the \(\ell_{\max}\) Apparent Discrepancy

A key two-stage distinction is emphasized:

1. **Orbital existence:** \(\ell_{\max}^{\mathrm{exists}}(n)\) from the inequality.
2. **Orbital filling:** determined by energy ordering + screening.

Thus, for \(n=3\), \(3d\) can exist mathematically but fills after \(4s\), so period 3 remains \(3s,3p\).

## 5) Period Table Construction Recipe

1. Apply PCE constraint to derive admissible \((n,\ell)\).
2. Evaluate ordering functional with screening corrections.
3. Fill orbitals in Madelung-like order.
4. Group elements by period as encountered in the filling sequence.
5. Recover s/p/d/f block structure and period lengths.

## 6) Superheavy Boundary Claim

The framework associates the endpoint of chemical periodicity with relativistic criticality:

- Bohr/Dirac heuristic bound near \(Z\approx 1/\alpha\approx 137\)
- QED-extended critical region around \(Z\approx 173\)

and cites an island-of-stability target near \(Z=126, N=184\).

## 7) Compact Existence-vs-Filling Table

| Period | Highest occupied \(n\) | \(\ell_{\max}^{\mathrm{exists}}\) at that \(n\) | Orbitals filling during period |
|---|---:|---:|---|
| 1 | 1 | 0 | 1s |
| 2 | 2 | 1 | 2s, 2p |
| 3 | 3 | 2 | 3s, 3p |
| 4 | 4 | 3 | 4s, 3d, 4p |
| 5 | 5 | 3 | 5s, 4d, 5p |
| 6 | 6 | 4 | 6s, 4f, 5d, 6p |
| 7 | 7 | 4 | 7s, 5f, 6d, 7p |

## 8) Notes

- This document is a faithful consolidation of the provided manuscript narrative.
- It preserves the stated invariants and two-stage logic (existence vs filling) while removing duplicate pasted sections.
