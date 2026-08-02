# A/B Testing & Experimentation Interview Questions

## Table of Contents
- [Experiment Setup & Hypothesis Formulation](#experiment-setup--hypothesis-formulation)
- [Sample Size Calculation & Power Analysis](#sample-size-calculation--power-analysis)
- [Variance Reduction Techniques (CUPED)](#variance-reduction-techniques-cuped)
- [Network Effects & Interference](#network-effects--interference)
- [Common Pitfalls (Novelty Effect, Peeking, Multiple Testing)](#common-pitfalls-novelty-effect-peeking-multiple-testing)

---

## Experiment Setup & Hypothesis Formulation

### Q: What is the Minimum Detectable Effect (MDE) and how does it influence sample size?
<details>
<summary><b>💡 Show Answer</b></summary>

- **MDE**: The smallest relative or absolute lift in a key metric that an experiment is powered to detect with statistical significance (power $1-\beta = 0.80$).
- **Sample Size Relation**: Sample size $N \propto \frac{1}{\text{MDE}^2}$. Detecting smaller MDEs requires quadratically larger sample sizes.

</details>

---

## Variance Reduction Techniques (CUPED)

### Q: Explain CUPED (Controlled-Experiment Using Pre-Experiment Data) for A/B testing.
<details>
<summary><b>💡 Show Answer</b></summary>

CUPED reduces metric variance by removing predictable variation using pre-experiment feature $X$:


$$Y_{\text{CUPED}} = Y - \theta (X - \mathbb{E}[X]), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$


This increases statistical power and decreases required sample size without bias.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
