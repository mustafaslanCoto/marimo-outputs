# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def intro():
    import marimo as mo

    mo.md(r"""
    # Reinforcement Learning & MDPs — Curriculum Overview
    ### *Sequential Decision-Making, Bellman Operators, Dynamic Programming & RL*

    > **Primary Sources:** Puterman (*Markov Decision Processes*); Bertsekas (*Reinforcement Learning and Optimal Control*); Sutton & Barto (*Reinforcement Learning: An Introduction*); Powell (*RL & Stochastic Optimization*).

    ---

    ### Planned Modules (Coming Soon)
    1. **RL-01:** Foundations of Sequential Decision Processes
    2. **RL-02:** Finite-Horizon Markov Decision Processes & Backward Induction
    3. **RL-03:** Infinite-Horizon Discounted MDPs & Bellman Contraction Operators
    4. **RL-04:** Exact Dynamic Programming (Value Iteration, Policy Iteration, LP Formulation of MDPs)
    5. **RL-05:** Total-Reward & Average-Reward MDPs
    6. **RL-06:** Exploration vs. Exploitation & Multi-Armed Bandits
    7. **RL-07:** Tabular Model-Free RL (Monte Carlo, TD, SARSA, Q-Learning)
    8. **RL-08:** Value Function Approximation & Approximate Dynamic Programming
    9. **RL-09:** Policy Gradient & Actor-Critic Methods
    10. **RL-10:** Model-Based RL, Rollout & Lookahead Policies
    """)
    return (mo,)


if __name__ == "__main__":
    app.run()
