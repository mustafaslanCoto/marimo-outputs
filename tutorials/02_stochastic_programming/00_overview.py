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
    # Stochastic Programming — Curriculum Overview
    ### *Optimization Under Uncertainty, Recourse Models, and Decomposition*

    > **Primary Sources:** Birge & Louveaux (*Introduction to Stochastic Programming*); Shapiro, Dentcheva & Ruszczyński (*Lectures on Stochastic Programming*).

    ---

    ### Planned Modules (Coming Soon)
    1. **SP-01:** Decision-Making Under Uncertainty — Wait-and-See vs. Here-and-Now, Non-Anticipativity
    2. **SP-02:** Two-Stage Stochastic Linear Programming with Recourse
    3. **SP-03:** Analytical Properties of Recourse Functions
    4. **SP-04:** Value of Information ($EVPI$) & Value of the Stochastic Solution ($VSS$)
    5. **SP-05:** Chance-Constrained Programming
    6. **SP-06:** Decomposition Algorithms — The L-Shaped Method (Stochastic Benders)
    7. **SP-07:** Sample Average Approximation (SAA)
    8. **SP-08:** Multistage Stochastic Programming & Scenario Trees (SDDP)
    9. **SP-09:** Risk-Averse Optimization ($VaR$, $CVaR$)
    """)
    return (mo,)


if __name__ == "__main__":
    app.run()
