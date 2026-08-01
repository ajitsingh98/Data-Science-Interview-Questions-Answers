# Reinforcement Learning Interview Questions

## Table of Contents
- [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
- [Value-Based Methods (Q-Learning, DQN)](#value-based-methods-q-learning-dqn)
- [Policy-Based Methods (Policy Gradients, PPO)](#policy-based-methods-policy-gradients-ppo)
- [Multi-Armed Bandits & Exploration vs Exploitation](#multi-armed-bandits--exploration-vs-exploitation)

---

## Markov Decision Processes (MDPs)

### Q: Define a Markov Decision Process (MDP) and the Bellman Optimality Equation.
<details>
<summary><b>💡 Show Answer</b></summary>

An MDP is defined by tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.
- **Bellman Optimality Equation**:
  $$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
