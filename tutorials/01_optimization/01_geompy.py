# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "marimo-lens>=0.2.3",
#     "matplotlib",
#     "mcp>=2.0.0",
#     "nbformat==5.11.1",
#     "numpy",
#     "openai==3.51.0",
#     "pandas==3.1.1",
#     "polars",
#     "pytest==9.2.0",
#     "python-lsp-ruff==2.3.4",
#     "python-lsp-server==1.15.0",
#     "ruff==0.17.5",
#     "scipy",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "websockets==17.4",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from scipy.optimize import linprog
    import matplotlib.pyplot as plt
    import polars as pl
    import pandas as pd

    return linprog, mo, np, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 01: The Geometry of Linear Programming
    """)
    return


@app.cell(hide_code=True)
def header_intro(mo):
    mo.md(r"""
    ### *From Hyperplanes and Polyhedra to Basic Feasible Solutions, Degeneracy, and the Fundamental Theorem of LP*

    ---

    ### Why This Chapter Is the Heart of Linear Optimization

    Why can an optimization problem with **continuous variables** (infinitely many feasible points) be solved in a **finite number of steps** by checking only "corner points"? And how does a computer—which cannot "see" a 10,000-dimensional geometric corner—recognize a corner using only matrix algebra (`Ax = b`)?

    > **Quick Intuition — What is a *Feasible Point*?**
    > A **feasible point** is any assignment of values to the decision variables $x \in \mathbb{R}^n$ that satisfies **every** constraint of the optimization problem simultaneously (such as $Ax \ge b$, or $Ax = b$ with $x \ge 0$). The set of all feasible points together forms the **feasible region** (our polyhedron $P$).

    The entire power of Linear Programming rests on a **three-way bridge** between:

    1. **Geometry (Convex Analysis):** *Polyhedra*, *Extreme Points* (points that cannot be written as strict midpoints/averages of two other feasible points), and *Vertices* (unique minimizers of some linear cost).
    2. **Linear Algebra (Simplex Mechanics):** *Basic Feasible Solutions (BFS)*—points where $n$ linearly independent constraints are active (tight) simultaneously, or equivalently where $m$ linearly independent columns $B = [A_{B(1)} \cdots A_{B(m)}]$ synthesize the right-hand side $b$ via $x_B = B^{-1}b \ge 0$.
    3. **Optimization Theory:** Even when a polyhedron contains uncountably many points, if an optimal solution exists and the polyhedron contains no infinite lines, **at least one optimal solution is guaranteed to sit at an extreme point (BFS)**—meaning it is a *mathematical certainty* (never a heuristic or lucky accident) that an algorithm searching only corners will never miss the global optimum.

    Every section below follows our **Four-Pillar Pedagogical Structure**:

    1. **Formal Theory & Proofs** (exact definitions, theorems, and proofs)
    2. **Plain-English Deconstruction & Notation Lexicon** (translating every symbol, transpose, and condition into clear English)
    3. **Worked Analytical Example** (hand-worked step-by-step calculation)
    4. **Interactive Python & `scipy.optimize.linprog` (HiGHS) Laboratory** (executable code and visual intuition)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Polyhedra and Convex Sets
    """)
    return


@app.cell(hide_code=True)
def sec1_theory(mo):
    mo.md(r"""
    ### 1.1 Formal Theory & Mathematical Definitions

    In Linear Programming, we optimize a linear objective over a feasible region defined by linear equality and inequality constraints. Geometrically, what kind of object does a system of linear inequalities carve out of $\mathbb{R}^n$?

    #### **Definition 2.1 (Polyhedron)**
    A **polyhedron** is a subset $P \subseteq \mathbb{R}^n$ described by a finite number of linear inequalities:

    $$
    P = \{x \in \mathbb{R}^n \mid Ax \ge b\}
    $$

    where $A \in \mathbb{R}^{m \times n}$ is an $m \times n$ matrix and $b \in \mathbb{R}^m$ is an $m$-dimensional vector.

    > **Why all LP feasible sets fit this definition:** Any equality constraint $a_i'x = b_i$ can be written as two opposing inequalities ($a_i'x \ge b_i$ and $-a_i'x \ge -b_i$), and any $\le$ constraint $a_i'x \le b_i$ can be multiplied by $-1$ to become $-a_i'x \ge -b_i$. Thus, **every linear programming feasible set is a polyhedron**. In particular, $P = \{x \in \mathbb{R}^n \mid Ax = b,\, x \ge 0\}$ is called a **polyhedron in standard form**.

    #### **Definition 2.2 (Bounded Set / Polytope)**
    A set $S \subseteq \mathbb{R}^n$ is **bounded** if there exists a constant $K > 0$ such that the absolute value of every component of every $x \in S$ satisfies $|x_j| \le K$ for all $j = 1, \dots, n$ (equivalently, $\|x\|_\infty \le K$). A bounded polyhedron is often called a **polytope**.

    #### **Definition 2.3 (Hyperplane and Halfspace)**
    Let $a \in \mathbb{R}^n$ be a **nonzero** vector ($a \ne 0$) and let $b \in \mathbb{R}$ be a scalar.

    1. **Hyperplane:** $H = \{x \in \mathbb{R}^n \mid a'x = b\}$
    2. **Closed Halfspace:** $H^+ = \{x \in \mathbb{R}^n \mid a'x \ge b\}$ (or $H^- = \{x \in \mathbb{R}^n \mid a'x \le b\}$)

    * **Geometric Orthogonality of $a$:** Why is the coefficient vector $a$ always **perpendicular (normal)** to the hyperplane $H$? Take any two points $x, y \in H$. Then $a'x = b$ and $a'y = b$. Subtracting the two gives:

    $$
    a'(x - y) = 0
    $$

    Since $x - y$ is any direction vector lying flat inside the hyperplane $H$, the vector $a$ is orthogonal to the hyperplane and points in the direction where $a'x$ increases ($a'x > b$).

    * **Polyhedron as an Intersection of Halfspaces:** If $a_1', \dots, a_m'$ denote the $m$ rows of $A$, then:

    $$
    P = \{x \in \mathbb{R}^n \mid Ax \ge b\} = \bigcap_{i=1}^m \{x \in \mathbb{R}^n \mid a_i'x \ge b_i\}
    $$

    Each single constraint slices $\mathbb{R}^n$ with a flat hyperplane and keeps one side (a halfspace). A polyhedron is simply what remains after making $m$ flat planar cuts.

    #### **Definition 2.4 & 2.5 (Convex Set, Convex Combination, and Convex Hull)**
    * **Convex Set:** A set $S \subseteq \mathbb{R}^n$ is **convex** if for any two points $x, y \in S$ and any scalar $\lambda \in [0, 1]$:

    $$
    \lambda x + (1 - \lambda)y \in S
    $$

    Geometrically, the entire straight line segment connecting $x$ and $y$ stays inside $S$.

    * **Convex Combination:** Given $k$ vectors $x^1, \dots, x^k \in \mathbb{R}^n$ and scalars $\lambda_1, \dots, \lambda_k \ge 0$ such that $\sum_{i=1}^k \lambda_i = 1$, the weighted average

    $$
    x = \sum_{i=1}^k \lambda_i x^i
    $$

    is called a **convex combination** of $x^1, \dots, x^k$.

    * **Convex Hull:** The **convex hull** $\operatorname{conv}(\{x^1, \dots, x^k\})$ is the set of all convex combinations of $x^1, \dots, x^k$ (physically: if you hammer nails at $x^1, \dots, x^k$ and stretch a rubber band around them, the region enclosed is the convex hull).

    #### **Theorem 2.1 (Fundamental Properties of Convexity)**
    1. **(a)** The intersection $\bigcap_{i \in I} S_i$ of any collection of convex sets is convex.
    2. **(b)** **Every polyhedron is a convex set.**

    *Proof of (b):* Take a single halfspace $H_i = \{x \mid a_i'x \ge b_i\}$. If $x, y \in H_i$ and $\lambda \in [0, 1]$, then since $\lambda \ge 0$ and $1-\lambda \ge 0$:

    $$
    a_i'(\lambda x + (1-\lambda)y) = \lambda (a_i'x) + (1-\lambda)(a_i'y) \ge \lambda b_i + (1-\lambda)b_i = b_i
    $$

    So every halfspace is convex. Since a polyhedron $P = \bigcap_{i=1}^m H_i$ is the intersection of $m$ halfspaces, part (a) guarantees $P$ is convex. $\blacksquare$

    3. **(c)** A convex combination of a finite number of elements of a convex set also belongs to that set (by induction on $k$).
    4. **(d)** The convex hull of a finite number of vectors is a convex set.

    > **Crucial Consequence (*Every Local Minimum Is a Global Minimum*):**
    > Suppose $p \in P$ is a local minimum of $c'x$ over a polyhedron $P$, and suppose there were some other feasible point $q \in P$ with strictly lower cost $c'q < c'p$. Because $P$ is convex, every point on the line segment $x(\lambda) = \lambda p + (1-\lambda)q$ for $\lambda \in (0, 1)$ is feasible, and its cost is:
    >
    > $$c'x(\lambda) = \lambda c'p + (1-\lambda)c'q < c'p$$
    >
    > Even for $\lambda = 0.99999$ (arbitrarily close to $p$), $x(\lambda)$ has strictly lower cost than $p$, contradicting that $p$ is a local minimum! Thus, **LP cannot get trapped in non-global local minima**.

    ---

    ### 1.2 Plain-English Deconstruction & Notation Lexicon

    | Symbol / Expression | Reading in Plain English | What It Means Geometrically & Computationally |
    | :--- | :--- | :--- |
    | $x \in \mathbb{R}^n$ | "$x$ is a column vector in $n$-dimensional real space" | A candidate decision point $x = (x_1, x_2, \dots, x_n)'$ with $n$ decision variables. |
    | $a_i' x$ (or $a_i^\top x$) | "The inner (dot) product of row vector $a_i'$ with $x$" | We use prime ($'$) to denote matrix/vector **transpose**. Explicitly, $a_i'x = \sum_{j=1}^n a_{ij}x_j$. It measures how far $x$ projects along the normal direction $a_i$. |
    | $Ax \ge b$ | "Componentwise inequality between $Ax \in \mathbb{R}^m$ and $b \in \mathbb{R}^m$" | Shorthand for $m$ simultaneous inequalities: $a_1'x \ge b_1,\, a_2'x \ge b_2,\, \dots,\, a_m'x \ge b_m$. |
    | $\lambda x + (1-\lambda)y$, $\lambda \in [0, 1]$ | "Convex combination (interpolation) between $x$ and $y$" | Think of $\lambda$ as a **slider from $0$ to $1$**: at $\lambda = 0$ you stand at $y$; at $\lambda = 0.5$ you are at the exact midpoint $\frac{1}{2}(x+y)$; at $\lambda = 1$ you stand at $x$. |
    | $x^1, \dots, x^k$ | "Superscript-indexed vectors in $\mathbb{R}^n$" | **Watch out for superscripts vs. subscripts!** Throughout this notebook, $x^1, x^2, \dots, x^k$ denote $k$ *different vectors* in $\mathbb{R}^n$ (not powers!), whereas $x_1, x_2, \dots, x_n$ with *subscripts* denote the $n$ scalar coordinates of a single vector $x$. |

    ---

    ### 1.3 Worked Analytical Example

    Consider the 2D polyhedron $P \subset \mathbb{R}^2$ formed by the intersection of 4 halfspaces:

    1. $x_1 + x_2 \ge 2$ *(normal $a_1 = (1, 1)'$, $b_1 = 2$)*
    2. $-x_1 + 2x_2 \le 6 \iff x_1 - 2x_2 \ge -6$
    3. $x_1 \le 7 \iff -x_1 \ge -7$
    4. $x_2 \ge 1$

    Take two feasible points inside $P$: $p = (6, 2)'$ and $q = (1, 3)'$.

    * Check $p = (6, 2)'$: $6+2 = 8 \ge 2$ ✓; $6 - 2(2) = 2 \ge -6$ ✓; $6 \le 7$ ✓; $2 \ge 1$ ✓.
    * Check $q = (1, 3)'$: $1+3 = 4 \ge 2$ ✓; $1 - 2(3) = -5 \ge -6$ ✓; $1 \le 7$ ✓; $3 \ge 1$ ✓.

    Now form the convex combination $x(\lambda) = \lambda p + (1-\lambda)q$ for any $\lambda \in [0, 1]$:

    $$
    x(\lambda) = \begin{bmatrix} 6\lambda + 1(1-\lambda) \\ 2\lambda + 3(1-\lambda) \end{bmatrix} = \begin{bmatrix} 1 + 5\lambda \\ 3 - \lambda \end{bmatrix}
    $$

    At $\lambda = 0.5$ (midpoint): $x(0.5) = (3.5, 2.5)'$. Notice how the left-hand side of constraint 1 along the segment is:

    $$
    x_1(\lambda) + x_2(\lambda) = (1 + 5\lambda) + (3 - \lambda) = 4 + 4\lambda = \lambda(8) + (1-\lambda)(4) \ge \lambda(2) + (1-\lambda)(2) = 2
    $$

    Every constraint value at $x(\lambda)$ is literally the $\lambda$-weighted average of its values at $p$ and $q$!
    """)
    return


@app.cell
def sec1_ui(mo):
    lam_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.4,
        label=r"Convex combination weight $\lambda$ in $x(\lambda) = \lambda p + (1-\lambda)q$:",
        full_width=True,
    )
    lam_slider
    return (lam_slider,)


@app.cell
def sec1_code(lam_slider, linprog, mo, np, pl, plt):
    # Interactive Python + SciPy (HiGHS) Verification of Halfspace Intersection & Convexity
    # Polyhedron P: A x >= b in R^2
    A_sec1 = np.array([
        [ 1.0,  1.0],   # x1 + x2 >= 2
        [ 1.0, -2.0],   # x1 - 2*x2 >= -6  (i.e. -x1 + 2*x2 <= 6)
        [-1.0,  0.0],   # -x1 >= -7        (i.e. x1 <= 7)
        [ 0.0,  1.0],   # x2 >= 1
    ])
    b_sec1 = np.array([2.0, -6.0, -7.0, 1.0])
    c_sec1 = np.array([1.0, 2.0])  # Objective: minimize x1 + 2*x2

    p_pt = np.array([6.0, 2.0])
    q_pt = np.array([1.0, 3.0])
    lam_val = lam_slider.value
    x_lam = lam_val * p_pt + (1.0 - lam_val) * q_pt

    # Use scipy.optimize.linprog (HiGHS) to solve min c'x over P (A_ub = -A_sec1, b_ub = -b_sec1)
    _res1 = linprog(c_sec1, A_ub=-A_sec1, b_ub=-b_sec1, bounds=(None, None), method="highs")
    x_opt_sec1 = _res1.x.copy()
    obj_opt_sec1 = float(_res1.fun)

    # Compute surpluses s(x) = A x - b >= 0 at p, q, and x(lambda)
    surplus_p = A_sec1 @ p_pt - b_sec1
    surplus_q = A_sec1 @ q_pt - b_sec1
    surplus_lam = A_sec1 @ x_lam - b_sec1

    # Plot the 4 halfspaces, their normal vectors a_i, the line segment, and optimal corner
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=110)
    _x1_grid = np.linspace(-1.2, 8.0, 400)

    # Boundary lines
    ax1.plot(_x1_grid, 2.0 - _x1_grid, label=r"$a_1'x = 2\;\;(x_1 + x_2 \geq 2)$", lw=2, color="#2563eb")
    ax1.plot(_x1_grid, (_x1_grid + 6.0) / 2.0, label=r"$a_2'x = -6\;\;(x_1 - 2x_2 \geq -6)$", lw=2, color="#7c3aed")
    ax1.axvline(7.0, label=r"$a_3'x = -7\;\;(-x_1 \geq -7)$", lw=2, color="#db2777")
    ax1.axhline(1.0, label=r"$a_4'x = 1\;\;(x_2 \geq 1)$", lw=2, color="#059669")

    # Feasible polygon vertices
    _poly_pts = np.array([[1.0, 1.0], [7.0, 1.0], [7.0, 6.5], [-2.0/3.0, 8.0/3.0]])
    ax1.fill(_poly_pts[:, 0], _poly_pts[:, 1], color="#dbeafe", alpha=0.55, label=r"Polyhedron $P = \bigcap_{i=1}^4 H_i$")

    # Plot inward normal vectors a_i (scaled for visibility)
    _normals_anchor = [
        (0.2, 1.8, A_sec1[0] * 0.45, "#2563eb"),
        (2.0, 4.0, A_sec1[1] * 0.35, "#7c3aed"),
        (7.0, 3.8, A_sec1[2] * 0.65, "#db2777"),
        (4.0, 1.0, A_sec1[3] * 0.65, "#059669"),
    ]
    for _ox, _oy, _dv, _col in _normals_anchor:
        ax1.annotate("", xy=(_ox + _dv[0], _oy + _dv[1]), xytext=(_ox, _oy),
                     arrowprops=dict(arrowstyle="->", color=_col, lw=1.8))

    # Plot p, q, segment, x(lambda), and optimal solution
    ax1.plot([q_pt[0], p_pt[0]], [q_pt[1], p_pt[1]], "k--", lw=1.8, label="Segment $[q, p]$")
    ax1.scatter(*p_pt, color="#1e293b", s=70, zorder=5)
    ax1.text(p_pt[0] + 0.15, p_pt[1] - 0.15, "$p=(6,2)$", fontsize=9.5, weight="bold")
    ax1.scatter(*q_pt, color="#1e293b", s=70, zorder=5)
    ax1.text(q_pt[0] + 0.15, q_pt[1] + 0.15, "$q=(1,3)$", fontsize=9.5, weight="bold")
    ax1.scatter(*x_lam, color="#dc2626", s=110, zorder=6, label=rf"$x(\lambda) = ({x_lam[0]:.2f}, {x_lam[1]:.2f})$")
    ax1.scatter(*x_opt_sec1, color="#f59e0b", edgecolor="black", s=140, marker="*", zorder=7,
                label=rf"Optimal Solution $\min (x_1+2x_2)$ at $({x_opt_sec1[0]:.1f}, {x_opt_sec1[1]:.1f})$")

    ax1.set_xlim(-1.2, 8.2)
    ax1.set_ylim(0.2, 7.0)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_title(r"Polyhedron $P = \{x \in \mathbb{R}^2 \mid Ax \geq b\}$, Normal Vectors $a_i$, and Convexity")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    _df_sec1 = pl.DataFrame({
        "Constraint i": ["1: x1 + x2 >= 2", "2: x1 - 2x2 >= -6", "3: -x1 >= -7", "4: x2 >= 1"],
        "Surplus at q=(1,3)": surplus_q.round(3),
        "Surplus at p=(6,2)": surplus_p.round(3),
        f"Surplus at x(λ={lam_val:.2f})": surplus_lam.round(3),
        "Feasible (>= 0)?": ["Yes ✓" if s >= -1e-9 else "No" for s in surplus_lam],
    })

    mo.hstack([fig1, mo.vstack([mo.md(f"**Convex Combination:** $x({lam_val:.2f}) = ({x_lam[0]:.2f},\\, {x_lam[1]:.2f})' \\in P$\n\n**Optimal Corner:** $x^* = ({x_opt_sec1[0]:.1f},\\, {x_opt_sec1[1]:.1f})'$, $c'x^* = {obj_opt_sec1:.1f}$"), _df_sec1])], widths=[0.58, 0.42])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Extreme Points, Vertices, and Basic Feasible Solutions
    """)
    return


@app.cell(hide_code=True)
def sec2_theory(mo):
    mo.md(r"""
    ### 2.1 Formal Theory: Three Faces of a "Corner Point"

    Intuitively, we know what a "corner" of a 2D polygon or 3D polyhedron looks like. However, in $\mathbb{R}^n$, we need a rigorous definition. We introduce **three distinct definitions** of a corner—two geometric and one algebraic—and prove in **Theorem 2.3** that all three describe the exact same set of points.

    #### **1. Convex Geometry View — Definition 2.6 (Extreme Point)**
    Let $P$ be a polyhedron. A vector $x \in P$ is an **extreme point** of $P$ if we **cannot** find two vectors $y, z \in P$, both different from $x$ ($y \ne x, z \ne x$), and a scalar $\lambda \in [0, 1]$ such that:

    $$
    x = \lambda y + (1 - \lambda)z
    $$

    * **Midpoint Formulation:** Equivalently, $x \in P$ is an extreme point if it is **not the midpoint** $\frac{1}{2}(y + z)$ of any two distinct points $y, z \in P$. Notice this definition is **purely intrinsic to the set $P$**—it does not even mention matrices $A$ or vectors $b$.

    #### **2. Optimization / Supporting Hyperplane View — Definition 2.7 (Vertex)**
    Let $P$ be a polyhedron. A vector $x \in P$ is a **vertex** of $P$ if there exists some cost vector $c \in \mathbb{R}^n$ such that:

    $$
    c'x < c'y \quad \text{for all } y \in P \text{ with } y \ne x
    $$

    * **Geometric Meaning:** $x$ is the **unique optimal solution** to the linear program $\min_{y \in P} c'y$. Equivalently, the hyperplane $H = \{y \in \mathbb{R}^n \mid c'y = c'x\}$ slices space so that $P$ lies entirely on one side ($c'y \ge c'x$) and touches $H$ at the **single point** $\{x\}$ (a *strictly supporting hyperplane*).

    #### **3. Algebraic / Linear System View — Definitions 2.8 & 2.9 (Active Constraints & Basic Feasible Solution)**
    Neither geometric definition tells a computer *how to solve for a corner*. For that, we look at the linear constraints defining $P$:

    $$
    a_i'x \ge b_i \quad (i \in M_1), \qquad a_i'x \le b_i \quad (i \in M_2), \qquad a_i'x = b_i \quad (i \in M_3)
    $$

    Here $M_1, M_2, M_3$ are simply the **constraint index sets** partitioning all $m$ linear constraints by type ($\ge$, $\le$, and $=$), so $M = M_1 \cup M_2 \cup M_3$ is the set of all constraint indices.

    * **Definition 2.8 (Active / Binding Constraint):** If a vector $x^* \in \mathbb{R}^n$ satisfies $a_i'x^* = b_i$ with **equality** for some $i \in M_1 \cup M_2 \cup M_3$, we say constraint $i$ is **active** (or **binding**) at $x^*$. We denote the set of active constraint indices at $x^*$ by:

    $$
    I(x^*) = \{i \in M_1 \cup M_2 \cup M_3 \mid a_i'x^* = b_i\}
    $$

    * **Theorem 2.2 (When Active Constraints Pin Down a Unique Point):** Let $x^* \in \mathbb{R}^n$ have active index set $I = I(x^*)$. The following three statements are equivalent:
    1. **(a)** There exist **$n$ vectors** in the active normal family $\{a_i \mid i \in I\}$ that are **linearly independent**.
    2. **(b)** The span of $\{a_i \mid i \in I\}$ is all of $\mathbb{R}^n$: $\operatorname{span}(\{a_i \mid i \in I\}) = \mathbb{R}^n$.
    3. **(c)** The system of active equations $a_i'x = b_i$ ($i \in I$) has a **unique solution** in $\mathbb{R}^n$ (namely $x^*$).

    *Why?* If the active normals $\{a_i\}_{i \in I}$ do not span $\mathbb{R}^n$, their orthogonal complement is nontrivial—meaning there exists a nonzero vector $d \ne 0$ such that $a_i'd = 0$ for all $i \in I$. Then $a_i'(x^* + d) = a_i'x^* + 0 = b_i$ for all $i \in I$, so you can slide along $d$ without leaving any active hyperplane!

    * **Definition 2.9 (Basic Solution and Basic Feasible Solution):** Given a polyhedron $P \subseteq \mathbb{R}^n$:
    1. **(a) Basic Solution:** $x^* \in \mathbb{R}^n$ is a **basic solution** if (i) all equality constraints ($i \in M_3$) are satisfied ($a_i'x^* = b_i$), and (ii) out of the constraints active at $x^*$, there are **$n$ linearly independent** constraints.
    2. **(b) Basic Feasible Solution (BFS):** If $x^*$ is a basic solution that satisfies **all** constraints ($x^* \in P$), it is a **basic feasible solution**.

    ---

    ### 2.2 The Grand Equivalence Theorem & Proof Mechanics

    #### **Theorem 2.3 (Equivalence of Vertex, Extreme Point, and BFS)**
    Let $P = \{x \in \mathbb{R}^n \mid Ax \ge b\}$ be a nonempty polyhedron and let $x^* \in P$. Then the following are equivalent:

    $$
    \boxed{\text{(a) } x^* \text{ is a Vertex} \quad \Longleftrightarrow \quad \text{(b) } x^* \text{ is an Extreme Point} \quad \Longleftrightarrow \quad \text{(c) } x^* \text{ is a Basic Feasible Solution (BFS)}}
    $$

    Let us trace all three steps of the circular proof—because each step teaches a fundamental geometric technique:

    **Step 1: $\text{(a) Vertex} \implies \text{(b) Extreme Point}$**

    Suppose $x^*$ is a vertex. Then there exists $c \in \mathbb{R}^n$ such that $c'x^* < c'y$ for all $y \in P \setminus \{x^*\}$. Take any two points $y, z \in P$ with $y \ne x^*, z \ne x^*$ and any $\lambda \in [0, 1]$. Since $c'x^* < c'y$ and $c'x^* < c'z$, taking their convex combination gives:

    $$
    c'(\lambda y + (1-\lambda)z) = \lambda c'y + (1-\lambda)c'z > \lambda c'x^* + (1-\lambda)c'x^* = c'x^*
    $$

    Because $c'(\lambda y + (1-\lambda)z) > c'x^*$, the point $\lambda y + (1-\lambda)z$ **cannot equal $x^*$**. Thus $x^*$ is an extreme point.

    **Step 2: $\text{(b) Extreme Point} \implies \text{(c) Basic Feasible Solution}$ *(Contrapositive: $\neg\text{BFS} \implies \neg\text{Extreme Point}$)* **

    Suppose $x^* \in P$ is **not** a BFS. Let $I = \{i \mid a_i'x^* = b_i\}$ be the active constraints at $x^*$:

    * **Active constraints ($i \in I$):** Satisfied with equality, $a_i'x^* = b_i$.
    * **Inactive constraints ($i \notin I$):** Satisfied with **strict inequality** (strictly positive slack), $a_i'x^* > b_i$.

    Because $x^*$ is not a BFS, fewer than $n$ vectors in $\{a_i \mid i \in I\}$ are linearly independent. Therefore, $\{a_i \mid i \in I\}$ spans a proper subspace of $\mathbb{R}^n$, which means there exists a **nonzero direction vector $d \ne 0$ orthogonal to every active normal**:

    $$
    a_i'd = 0 \quad \text{for all } i \in I
    $$

    Now perturb $x^*$ forward and backward along $d$ by a tiny step $\epsilon > 0$:

    $$
    y = x^* + \epsilon d, \qquad z = x^* - \epsilon d
    $$

    * What happens to **active constraints ($i \in I$)**? $a_i'y = a_i'x^* + \epsilon(a_i'd) = b_i + 0 = b_i$, and similarly $a_i'z = b_i$. They stay right on the active boundaries!
    * What happens to **inactive constraints ($i \notin I$)**? Since $a_i'x^* - b_i > 0$, as long as we pick $\epsilon > 0$ small enough that $\epsilon |a_i'd| \le a_i'x^* - b_i$ for all $i \notin I$, both $y$ and $z$ still satisfy $a_i'y \ge b_i$ and $a_i'z \ge b_i$!

    Thus $y, z \in P$, $y \ne x^*$, $z \ne x^*$, and:

    $$
    x^* = \frac{1}{2}y + \frac{1}{2}z
    $$

    So $x^*$ is the midpoint of $y$ and $z$ and is **not** an extreme point!

    **Step 3: $\text{(c) Basic Feasible Solution} \implies \text{(a) Vertex}$ *(Constructing the Exact Objective Vector $c$)* **

    Suppose $x^* \in P$ is a BFS with active set $I = \{i \mid a_i'x^* = b_i\}$. How do we construct a cost vector $c$ that is uniquely minimized at $x^*$? Simply **sum the active inward normals**:

    $$
    c = \sum_{i \in I} a_i
    $$

    For any feasible point $x \in P$, we have $a_i'x \ge b_i$ for every $i \in I$. Summing over $i \in I$:

    $$
    c'x = \sum_{i \in I} a_i'x \ge \sum_{i \in I} b_i = \sum_{i \in I} a_i'x^* = c'x^*
    $$

    When can equality $c'x = c'x^*$ hold? Since every term satisfies $a_i'x - b_i \ge 0$, their sum is zero **if and only if every single term is zero**: $a_i'x = b_i$ for all $i \in I$. By Theorem 2.2, because $I$ contains $n$ linearly independent vectors, $x^*$ is the **unique** solution to $\{a_i'x = b_i,\, i \in I\}$. Hence $c'x > c'x^*$ for all $x \in P \setminus \{x^*\}$, proving $x^*$ is a vertex! $\blacksquare$

    #### **Corollary 2.1 (Finite Number of Basic Feasible Solutions)**
    For a polyhedron defined by $m$ linear constraints in $\mathbb{R}^n$, every basic solution is uniquely determined by choosing $n$ linearly independent active constraints out of $m$. Therefore:

    $$
    \#(\text{Extreme Points / BFSs}) \le \#(\text{Basic Solutions}) \le \binom{m}{n} = \frac{m!}{n!(m-n)!} < \infty
    $$

    * **Adjacent Basic Solutions & Edges:** Two distinct basic solutions in $\mathbb{R}^n$ are **adjacent** if they share $n - 1$ linearly independent active constraints. If two adjacent basic solutions are both feasible (BFSs), the line segment joining them is an **edge** of the polyhedron.

    ---

    ### 2.3 Plain-English Deconstruction & Notation Lexicon

    | Symbol / Concept | Plain-English Meaning | Why It Matters |
    | :--- | :--- | :--- |
    | $I(x^*) = \{i \mid a_i'x^* = b_i\}$ | **Active Set at $x^*$:** The list of constraints that have **zero slack** ($a_i'x^* - b_i = 0$) at $x^*$. | Tells you which walls of the polyhedron you are physically touching when standing at $x^*$. |
    | $\text{rank}(\{a_i\}_{i \in I(x^*)}) = n$ | **Full-Dimensional Corner Lock:** Out of the walls you are touching, $n$ of them point in linearly independent directions. | In $\mathbb{R}^2$, touching 1 wall leaves 1 degree of freedom (sliding along an edge); touching 2 non-parallel walls locks you into a 0-dimensional point (a corner). In $\mathbb{R}^n$, you need $n$ independent walls to lock all $n$ degrees of freedom. |
    | $a_i'd = 0 \;\; (\forall i \in I)$ | **Nullspace Perturbation Direction:** A direction $d \ne 0$ perpendicular to all active constraint normals. | If you touch fewer than $n$ independent walls, such a direction $d$ always exists, allowing you to wiggle forward ($+\epsilon d$) and backward ($-\epsilon d$) while staying glued to every wall in $I$. |
    | **Basic Solution** vs. **Basic Feasible Solution (BFS)** | A **Basic Solution** locks $n$ independent walls ($a_i'x = b_i$), ignoring whether the remaining $m-n$ walls are violated. A **BFS** also obeys all remaining $m-n$ inequalities. | Basic solutions outside $P$ are intersections of constraint extensions outside the feasible region. Note: *BFS* is representation-independent (since $\text{BFS} \iff \text{Extreme Point}$), whereas *infeasible basic solutions* depend on which inequalities are written down. |

    ---

    ### 2.4 Worked Analytical Example

    Consider $P \subset \mathbb{R}^2$ ($n = 2$) with $m = 4$ constraints written in $Ax \ge b$ form:

    1. $x_1 \ge 0 \quad (a_1 = (1, 0)',\, b_1 = 0)$
    2. $x_2 \ge 0 \quad (a_2 = (0, 1)',\, b_2 = 0)$
    3. $-x_1 - x_2 \ge -4 \iff x_1 + x_2 \le 4 \quad (a_3 = (-1, -1)',\, b_3 = -4)$
    4. $-x_1 + x_2 \ge -2 \iff x_1 - x_2 \le 2 \quad (a_4 = (-1, 1)',\, b_4 = -2)$

    Since $n = 2$ and $m = 4$, there are at most $\binom{4}{2} = 6$ basic solutions:

    * **Pair $\{1, 2\}$ ($x_1 = 0, x_2 = 0$):** $x = (0, 0)'$. Check constraints 3 & 4: $0 \le 4$ ✓, $0 \le 2$ ✓ $\implies$ **BFS #1**. Active normals sum: $c = a_1 + a_2 = (1, 1)'$; minimizing $x_1 + x_2$ over $P$ has unique optimum $(0, 0)'$!
    * **Pair $\{2, 4\}$ ($x_2 = 0, x_1 - x_2 = 2$):** $x = (2, 0)'$. Check 1 & 3: $2 \ge 0$ ✓, $2 \le 4$ ✓ $\implies$ **BFS #2**. Active normals sum: $c = a_2 + a_4 = (0, 1)' + (-1, 1)' = (-1, 2)'$; minimizing $-x_1 + 2x_2$ over $P$ has unique optimum $(2, 0)'$!
    * **Pair $\{3, 4\}$ ($x_1 + x_2 = 4, x_1 - x_2 = 2$):** $x = (3, 1)'$. Check 1 & 2: $3 \ge 0$ ✓, $1 \ge 0$ ✓ $\implies$ **BFS #3**. Active normals sum: $c = a_3 + a_4 = (-2, 0)'$; minimizing $-2x_1$ over $P$ has unique optimum $(3, 1)'$!
    * **Pair $\{1, 3\}$ ($x_1 = 0, x_1 + x_2 = 4$):** $x = (0, 4)'$. Check 2 & 4: $4 \ge 0$ ✓, $0 - 4 = -4 \le 2$ ✓ $\implies$ **BFS #4**.
    * **Pair $\{2, 3\}$ ($x_2 = 0, x_1 + x_2 = 4$):** $x = (4, 0)'$. Check 4: $4 - 0 = 4 \not\le 2$ ✗ $\implies$ **Basic Solution, NOT Feasible!**
    * **Pair $\{1, 4\}$ ($x_1 = 0, x_1 - x_2 = 2$):** $x = (0, -2)'$. Check 2: $-2 \not\ge 0$ ✗ $\implies$ **Basic Solution, NOT Feasible!**
    """)
    return


@app.cell
def sec2_ui(mo):
    bfs_selector = mo.ui.dropdown(
        options={
            "BFS 1: x* = (0, 0)  [Active: {1, 2}]": 0,
            "BFS 2: x* = (2, 0)  [Active: {2, 4}]": 1,
            "BFS 3: x* = (3, 1)  [Active: {3, 4}]": 2,
            "BFS 4: x* = (0, 4)  [Active: {1, 3}]": 3,
        },
        value="BFS 3: x* = (3, 1)  [Active: {3, 4}]",
        label=r"Select a Basic Feasible Solution $x^*$ to test Theorem 2.3 ($c = \sum_{i \in I(x^*)} a_i$):"
    )
    bfs_selector
    return (bfs_selector,)


@app.cell
def sec2_code(bfs_selector, linprog, mo, np, pl, plt):
    # Interactive Theorem 2.3 Laboratory: Basic Solutions, BFSs, Supporting Hyperplanes & Perturbation
    from itertools import combinations

    A_sec2 = np.array([
        [ 1.0,  0.0],   # 1: x1 >= 0
        [ 0.0,  1.0],   # 2: x2 >= 0
        [-1.0, -1.0],   # 3: -x1 - x2 >= -4  (x1 + x2 <= 4)
        [-1.0,  1.0],   # 4: -x1 + x2 >= -2  (x1 - x2 <= 2)
    ])
    b_sec2 = np.array([0.0, 0.0, -4.0, -2.0])
    constr_names_sec2 = ["1: x1 >= 0", "2: x2 >= 0", "3: -x1 - x2 >= -4", "4: -x1 + x2 >= -2"]

    # Enumerate all C(4, 2) = 6 pairs of constraints
    basic_records = []
    bfs_list = []
    infeas_bs_list = []

    for _idx_pair in combinations(range(4), 2):
        _A_sub = A_sec2[list(_idx_pair), :]
        _b_sub = b_sec2[list(_idx_pair)]
        if np.linalg.matrix_rank(_A_sub) == 2:
            _x_bs = np.linalg.solve(_A_sub, _b_sub)
            _slacks = A_sec2 @ _x_bs - b_sec2
            _is_feas = bool(np.all(_slacks >= -1e-9))
            basic_records.append({
                "Active Pair": f"{{{_idx_pair[0]+1}, {_idx_pair[1]+1}}}",
                "Point x*": f"({_x_bs[0]:.1f}, {_x_bs[1]:.1f})",
                "Min Surplus min(Ax - b)": float(np.min(_slacks)),
                "Classification": "Basic Feasible Solution (BFS / Extreme Point)" if _is_feas else "Basic Solution (Infeasible)",
            })
            if _is_feas:
                bfs_list.append((_x_bs, _idx_pair))
            else:
                infeas_bs_list.append((_x_bs, _idx_pair))

    # Order bfs_list to match dropdown: (0,0), (2,0), (3,1), (0,4)
    _bfs_ordered = [
        (np.array([0.0, 0.0]), (0, 1)),
        (np.array([2.0, 0.0]), (1, 3)),
        (np.array([3.0, 1.0]), (2, 3)),
        (np.array([0.0, 4.0]), (0, 2)),
    ]
    _sel_x, _sel_I = _bfs_ordered[bfs_selector.value]

    # Construct c = sum_{i in I(x*)} a_i as in the proof of Theorem 2.3 (BFS => Vertex)
    c_thm23 = A_sec2[list(_sel_I), :].sum(axis=0)

    # Solve min c'x s.t. A x >= b with SciPy (HiGHS) to verify x* is the unique optimal vertex
    _res2 = linprog(c_thm23, A_ub=-A_sec2, b_ub=-b_sec2, bounds=(None, None), method="highs")
    opt_vertex = _res2.x.copy()
    opt_val_sec2 = float(_res2.fun)

    # Also illustrate non-BFS point w = (1.5, 2.5) on constraint 3 only (rank 1 < n=2)
    w_non_bfs = np.array([1.5, 2.5])  # Active only at constraint 3: -x1 - x2 = -4
    d_null = np.array([1.0, -1.0])    # Orthogonal to a_3 = (-1, -1): a_3' d = 0
    eps_step = 0.8
    y_pert = w_non_bfs + eps_step * d_null
    z_pert = w_non_bfs - eps_step * d_null

    fig2, ax2 = plt.subplots(figsize=(7.6, 5.0), dpi=110)
    _xg = np.linspace(-1.0, 5.0, 400)
    ax2.axvline(0.0, color="#64748b", lw=1.5, ls="-", label="1: $x_1 = 0$")
    ax2.axhline(0.0, color="#64748b", lw=1.5, ls="-.", label="2: $x_2 = 0$")
    ax2.plot(_xg, 4.0 - _xg, color="#2563eb", lw=1.8, label="3: $x_1 + x_2 = 4$")
    ax2.plot(_xg, _xg - 2.0, color="#7c3aed", lw=1.8, label="4: $x_1 - x_2 = 2$")

    _poly2 = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [0.0, 4.0]])
    ax2.fill(_poly2[:, 0], _poly2[:, 1], color="#dcfce7", alpha=0.6, label="Feasible Polyhedron $P$")

    # Plot all BFSs and Infeasible Basic Solutions
    for _pt, _pair in _bfs_ordered:
        ax2.scatter(*_pt, color="#16a34a", edgecolor="black", s=90, zorder=5)
        ax2.text(_pt[0] + 0.1, _pt[1] + 0.12, f"BFS ({_pt[0]:.0f},{_pt[1]:.0f})", fontsize=8.5, weight="bold", color="#166534")

    for _pt, _pair in infeas_bs_list:
        ax2.scatter(*_pt, color="#ef4444", marker="x", s=90, lw=2.2, zorder=5)
        ax2.text(_pt[0] + 0.1, _pt[1] - 0.25, f"Infeasible BS ({_pt[0]:.0f},{_pt[1]:.0f})", fontsize=8.5, color="#991b1b")

    # Plot non-BFS point w and perturbation y = w + eps*d, z = w - eps*d
    ax2.plot([z_pert[0], y_pert[0]], [z_pert[1], y_pert[1]], color="#d97706", lw=2.5, zorder=6)
    ax2.scatter([z_pert[0], y_pert[0]], [z_pert[1], y_pert[1]], color="#d97706", s=55, zorder=6)
    ax2.scatter(*w_non_bfs, color="#f59e0b", edgecolor="black", s=85, zorder=7,
                label=r"Non-BFS $w=(1.5,2.5) = \frac{1}{2}(y+z)$")

    # Plot supporting hyperplane c'x = c'x* for the selected BFS
    _rhs_supp = float(c_thm23 @ _sel_x)
    if abs(c_thm23[1]) > 1e-8:
        _ysupp = (_rhs_supp - c_thm23[0] * _xg) / c_thm23[1]
        ax2.plot(_xg, _ysupp, color="#dc2626", lw=2.5, ls="--",
                 label=rf"Supporting Hyperplane $c'x = {_rhs_supp:.1f}$ ($c=({c_thm23[0]:.0f},{c_thm23[1]:.0f})'$)")
    else:
        ax2.axvline(_rhs_supp / c_thm23[0], color="#dc2626", lw=2.5, ls="--",
                    label=rf"Supporting Hyperplane $c'x = {_rhs_supp:.1f}$ ($c=({c_thm23[0]:.0f},{c_thm23[1]:.0f})'$)")

    ax2.scatter(*opt_vertex, color="#dc2626", edgecolor="black", s=160, marker="*", zorder=8,
                label=f"Optimal Solution (Unique Vertex): ({opt_vertex[0]:.1f}, {opt_vertex[1]:.1f})")

    ax2.set_xlim(-0.8, 4.8)
    ax2.set_ylim(-2.6, 4.8)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_title("Theorem 2.3: All 6 Basic Solutions (4 BFS Corners + 2 Infeasible BS) & Supporting Hyperplane")
    ax2.legend(loc="lower right", fontsize=7.8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.hstack([
        fig2,
        mo.vstack([
            mo.md(
                f"**Selected BFS:** $x^* = ({_sel_x[0]:.0f}, {_sel_x[1]:.0f})'$ with active set $I(x^*) = \\{{{_sel_I[0]+1}, {_sel_I[1]+1}\\}}$\n\n"
                f"**Constructed Cost Vector (Theorem 2.3):** $c = a_{_sel_I[0]+1} + a_{_sel_I[1]+1} = ({c_thm23[0]:.0f}, {c_thm23[1]:.0f})'$\n\n"
                f"**Optimal Solution to $\\min_{{x \\in P}} c'x$:** $x^* = ({opt_vertex[0]:.1f}, {opt_vertex[1]:.1f})'$ with objective $c'x^* = {opt_val_sec2:.1f}$"
            ),
            pl.DataFrame(basic_records),
        ])
    ], widths=[0.56, 0.44])
    return A_sec2, b_sec2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Polyhedra in Standard Form & The Simplex Algebra
    """)
    return


@app.cell(hide_code=True)
def sec3_theory(mo):
    mo.md(r"""
    ### 3.1 Formal Theory: From Active Constraints to Basis Matrices $B$

    While general polyhedra $\{x \in \mathbb{R}^n \mid Ax \ge b\}$ are convenient for geometric proofs, industrial LP solvers (such as HiGHS or GLPK) operate on **polyhedra in standard form**:

    $$
    P = \{x \in \mathbb{R}^n \mid Ax = b,\; x \ge 0\}
    $$

    where $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns ($m \le n$), and we assume $\operatorname{rank}(A) = m$ (the $m$ rows of $A$ are linearly independent).

    #### **How Does Definition 2.9 (Basic Solution) Specialize to Standard Form?**
    In standard form, there are $m$ equality constraints ($Ax = b$) and $n$ nonnegativity inequalities ($x_1 \ge 0, \dots, x_n \ge 0$), for a total of $m + n$ constraints in $\mathbb{R}^n$:

    1. Every basic solution **must satisfy all $m$ equality constraints $Ax = b$**, which immediately gives us **$m$ linearly independent active constraints**.
    2. To reach a total of **$n$ linearly independent active constraints** in $\mathbb{R}^n$, we must pick **$n - m$ variables and set them to zero** ($x_i = 0$), making their nonnegativity constraints $x_i \ge 0$ active!
    3. Which $n - m$ variables can we set to zero? If we set $n - m$ variables to zero, the remaining **$m$ variables** $x_{B(1)}, \dots, x_{B(m)}$ must be uniquely determined by the $m$ equations $Ax = b$. That happens **if and only if** their $m$ corresponding columns $A_{B(1)}, \dots, A_{B(m)}$ are linearly independent!

    #### **Theorem 2.4 (Characterization of Basic Solutions in Standard Form)**
    Consider the constraints $Ax = b$ and $x \ge 0$, where the $m \times n$ matrix $A$ has linearly independent rows ($\operatorname{rank}(A) = m$). A vector $x \in \mathbb{R}^n$ is a **basic solution** if and only if $Ax = b$ and there exist $m$ distinct indices $B(1), \dots, B(m) \in \{1, \dots, n\}$ such that:

    1. **(a)** The $m$ columns $A_{B(1)}, \dots, A_{B(m)} \in \mathbb{R}^m$ are **linearly independent**;
    2. **(b)** If $i \notin \{B(1), \dots, B(m)\}$, then $x_i = 0$.

    **Proof:**

    * **$({\Longleftarrow})$ Suppose (a) and (b) hold:** Since $x_i = 0$ for all $i \notin \{B(1), \dots, B(m)\}$, the system $Ax = b$ reduces to:

    $$
    \sum_{i=1}^m A_{B(i)} x_{B(i)} = b
    $$

    Because the $m$ columns $A_{B(1)}, \dots, A_{B(m)} \in \mathbb{R}^m$ are linearly independent, they form an invertible $m \times m$ matrix $B$, so $x_{B(1)}, \dots, x_{B(m)}$ are uniquely determined ($x_B = B^{-1}b$). Since the active constraints ($Ax = b$ and $x_i = 0$ for $i \notin B$) have a unique solution, Theorem 2.2 guarantees that $n$ of these active constraints are linearly independent. Thus $x$ is a basic solution.

    * **$({\Longrightarrow})$ Suppose $x$ is a basic solution:** Let $x_{B(1)}, \dots, x_{B(k)}$ be the strictly nonzero components of $x$. By Definition 2.9 and Theorem 2.2, the active constraints $\sum_{i=1}^k A_{B(i)} x_{B(i)} = b$ and $x_i = 0$ ($i \notin \{B(1), \dots, B(k)\}$) have a unique solution. Thus the columns $A_{B(1)}, \dots, A_{B(k)}$ must be linearly independent (otherwise a nontrivial combination $\sum_{i=1}^k \lambda_i A_{B(i)} = 0$ would yield multiple solutions $x_{B(i)} + \lambda_i$). Hence $k \le m$. Because $\operatorname{rank}(A) = m$, the column space of $A$ is $\mathbb{R}^m$, so we can extend the independent set $\{A_{B(1)}, \dots, A_{B(k)}\}$ by picking $m - k$ additional columns $A_{B(k+1)}, \dots, A_{B(m)}$ of $A$ to form $m$ linearly independent columns, with $x_{B(k+1)} = \dots = x_{B(m)} = 0$ and $x_i = 0$ for all $i \notin \{B(1), \dots, B(m)\}$. $\blacksquare$

    ---

    ### 3.2 Canonical 3-Step Procedure & Matrix Notation

    Every basic solution (and therefore every basic feasible solution) of $P = \{x \in \mathbb{R}^n \mid Ax = b, x \ge 0\}$ is constructed by:

    **Step 1 (Choose a Basis):** Choose $m$ linearly independent columns $A_{B(1)}, \dots, A_{B(m)}$ of $A$, forming the invertible $m \times m$ **basis matrix**:

    $$
    B = \begin{bmatrix} | & | & & | \\ A_{B(1)} & A_{B(2)} & \cdots & A_{B(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{m \times m}, \qquad \det(B) \ne 0
    $$

    **Step 2 (Fix Nonbasic Variables to Zero):** Set all $n - m$ **nonbasic variables** to zero:

    $$
    x_i = 0 \quad \text{for all } i \notin \{B(1), \dots, B(m)\} \qquad (x_N = 0)
    $$

    **Step 3 (Solve for Basic Variables):** Solve the $m \times m$ linear system $B x_B = b$ for the vector of **basic variables** $x_B = (x_{B(1)}, \dots, x_{B(m)})'$:

    $$
    \boxed{x_B = B^{-1}b}
    $$

    * **Feasibility Test:** If **$x_B = B^{-1}b \ge 0$** (every basic variable is nonnegative), then $x$ is a **Basic Feasible Solution (BFS)**! If even one component of $B^{-1}b$ is negative, $x$ is an *infeasible basic solution*.

    #### **Column Geometry ("Requirement Space") Intuition**
    Write $Ax = b$ as a linear combination of the columns of $A$:

    $$
    \sum_{j=1}^n A_j x_j = b \quad \xrightarrow{\text{set } x_N = 0} \quad A_{B(1)} x_{B(1)} + \dots + A_{B(m)} x_{B(m)} = b
    $$

    * **When do columns form a valid basis?** When $A_{B(1)}, \dots, A_{B(m)}$ span $\mathbb{R}^m$ (in $\mathbb{R}^2$, two columns that do not lie on the same line through the origin).
    * **When is the basis FEASIBLE ($B^{-1}b \ge 0$)?** When the target requirement vector $b \in \mathbb{R}^m$ lies inside the **nonnegative cone** $\operatorname{cone}(A_{B(1)}, \dots, A_{B(m)}) = \{\sum_{i=1}^m \theta_i A_{B(i)} \mid \theta_i \ge 0\}$ spanned by those $m$ columns!

    #### **Theorem 2.5 (Why Full Row Rank $\operatorname{rank}(A) = m$ Is Without Loss of Generality)**
    If $P = \{x \in \mathbb{R}^n \mid Ax = b, x \ge 0\}$ is **nonempty** and $\operatorname{rank}(A) = k < m$, let $a_{i_1}', \dots, a_{i_k}'$ be any $k$ linearly independent rows of $A$. Then every other row $a_i'$ is a linear combination $a_i' = \sum_{j=1}^k \lambda_{ij} a_{i_j}'$, and because $P \ne \emptyset$, $b_i = \sum_{j=1}^k \lambda_{ij} b_{i_j}$ is automatically consistent. Discarding the $m - k$ dependent rows leaves the **exact same polyhedron**:

    $$
    Q = \{x \in \mathbb{R}^n \mid a_{i_j}'x = b_{i_j},\; j = 1, \dots, k,\; x \ge 0\} = P
    $$

    ---

    ### 3.3 Plain-English Deconstruction & Notation Lexicon

    | Symbol / Term | Plain-English Meaning | Dimensions & Role |
    | :--- | :--- | :--- |
    | $B = \{B(1), \dots, B(m)\}$ | **Basis Index Set:** The subset of $m$ column indices chosen to be "in the basis". | Set of $m$ integers from $\{1, \dots, n\}$. Two bases are considered the *same* if they contain the same set of indices (regardless of ordering). |
    | $A_{B(i)}$ vs. $a_i'$ | **Column vs. Row Notation:** Uppercase $A_j \in \mathbb{R}^m$ is the **$j$-th column** of $A$; lowercase $a_i' \in \mathbb{R}^{1 \times n}$ is the **$i$-th row** of $A$. | We follow this capitalization convention systematically: $Ax = \sum_{j=1}^n A_j x_j$ (column view) vs. $(Ax)_i = a_i' x$ (row view). |
    | $B = [A_{B(1)} \cdots A_{B(m)}]$ | **Basis Matrix:** The $m \times m$ square submatrix formed by pasting the $m$ basic columns side by side. | Must be nonsingular ($\det(B) \ne 0 \iff \operatorname{rank}(B) = m$). |
    | $x_B = B^{-1}b$ | **Basic Variable Vector:** The unique values of the $m$ basic variables required to synthesize $b$. | Vector in $\mathbb{R}^m$. The full solution $x \in \mathbb{R}^n$ has $x_{B(i)} = (B^{-1}b)_i$ and $x_j = 0$ for $j \notin B$. |
    | **Adjacent Bases** | Two bases $B_1, B_2$ that share **$m - 1$ common basic indices** and differ in only 1 index. | One step (pivot) of the Simplex method swaps 1 column out of $B$ and 1 column in, moving between adjacent bases! |

    ---

    ### 3.4 Worked Analytical Example

    Consider a standard form polyhedron in $\mathbb{R}^7$ ($n = 7$ variables, $m = 4$ equality constraints) obtained by adding 4 slack variables $(x_4, x_5, x_6, x_7)$ to $x_1 + x_2 + 2x_3 \le 8$, $x_2 + 6x_3 \le 12$, $x_1 \le 4$, $x_2 \le 6$:

    $$
    \underbrace{\begin{bmatrix}
    1 & 1 & 2 & 1 & 0 & 0 & 0 \\
    0 & 1 & 6 & 0 & 1 & 0 & 0 \\
    1 & 0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 1
    \end{bmatrix}}_{A \in \mathbb{R}^{4 \times 7}} x = \underbrace{\begin{bmatrix} 8 \\ 12 \\ 4 \\ 6 \end{bmatrix}}_{b \in \mathbb{R}^4}, \qquad x \ge 0
    $$

    **Candidate Basis 1:** $\{B(1), B(2), B(3), B(4)\} = \{4, 5, 6, 7\}$ (the 4 slack columns).

    * Nonbasic variables set to zero: $x_1 = x_2 = x_3 = 0$.
    * Basis matrix: $B_1 = [A_4\; A_5\; A_6\; A_7] = I_4$ (invertible!).
    * Basic variables: $x_B = B_1^{-1}b = (8, 12, 4, 6)' \ge 0$.
    * Full vector: $x^{(1)} = (0, 0, 0, 8, 12, 4, 6)' \ge 0 \implies$ **Basic Feasible Solution (BFS)!**

    **Candidate Basis 2 (Adjacent to Basis 1 — swap $A_4 \leftrightarrow A_3$):** $\{3, 5, 6, 7\}$.

    * Nonbasic variables set to zero: $x_1 = x_2 = x_4 = 0$.
    * System $B_2 x_B = b$:
    * Row 1: $2x_3 = 8 \implies x_3 = 4$
    * Row 2: $6x_3 + x_5 = 12 \implies 24 + x_5 = 12 \implies x_5 = -12 < 0$!
    * Row 3: $x_6 = 4$; Row 4: $x_7 = 6$.
    * Full vector: $x^{(2)} = (0, 0, 4, 0, -12, 4, 6)'$. Because $x_5 = -12 < 0$, this is a **Basic Solution, but INFEASIBLE**.
    """)
    return


@app.cell
def sec3_ui(mo):
    basis_dropdown_sec3 = mo.ui.dropdown(
        options={
            "Basis {4, 5, 6, 7} (Slacks I_4)": [3, 4, 5, 6],
            "Basis {3, 5, 6, 7} (Adjacent to {4,5,6,7})": [2, 4, 5, 6],
            "Basis {1, 2, 5, 6} (Optimal for max 3x1 + 2x2 + 4x3)": [0, 1, 4, 5],
            "Basis {1, 2, 3, 7} (Degenerate BFS)": [0, 1, 2, 6],
            "Columns {1, 4, 5, 6} (Singular / Linearly Dependent!)": [0, 3, 4, 5],
        },
        value="Basis {4, 5, 6, 7} (Slacks I_4)",
        label="Select 4 columns of A in the standard-form example (m=4, n=7):"
    )
    basis_dropdown_sec3
    return (basis_dropdown_sec3,)


@app.cell
def sec3_code(basis_dropdown_sec3, linprog, mo, np, pl, plt):
    # Interactive Standard-Form Basis Explorer + Column Geometry + Optimal Basis
    A_ex21 = np.array([
        [1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 6.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    b_ex21 = np.array([8.0, 12.0, 4.0, 6.0])
    c_ex21 = np.array([-3.0, -2.0, -4.0, 0.0, 0.0, 0.0, 0.0])  # min -3x1 - 2x2 - 4x3

    sel_cols = basis_dropdown_sec3.value
    B_cand = A_ex21[:, sel_cols]
    rank_B = int(np.linalg.matrix_rank(B_cand))
    det_B = float(np.linalg.det(B_cand))

    if rank_B == 4:
        x_B_val = np.linalg.solve(B_cand, b_ex21)
        x_full = np.zeros(7)
        x_full[sel_cols] = x_B_val
        is_bfs_sec3 = bool(np.all(x_B_val >= -1e-9))
        status_str = (
            "✅ **Basic Feasible Solution (BFS)** ($\\text{rank}(B)=4$ and $x_B = B^{-1}b \\ge 0$)"
            if is_bfs_sec3
            else "⚠️ **Infeasible Basic Solution** ($\\text{rank}(B)=4$, but $x_B = B^{-1}b$ has negative entries!)"
        )
    else:
        x_full = np.full(7, np.nan)
        status_str = "❌ **NOT a Basis!** ($\\text{rank}(B) < 4$, columns are linearly dependent / singular matrix)"

    # Solve with SciPy (HiGHS) and identify the optimal Simplex Basis (non-zero basic variables)
    _res3 = linprog(c_ex21, A_eq=A_ex21, b_eq=b_ex21, bounds=(0.0, None), method="highs")
    opt_x_ex21 = _res3.x.copy()
    opt_vbasis = [0 if val > 1e-7 else -1 for val in opt_x_ex21]
    opt_basis_ex21 = [i + 1 for i, vb in enumerate(opt_vbasis) if vb == 0]

    # Plot Column Geometry: m=2, n=4 in R^2
    fig3, ax3 = plt.subplots(figsize=(6.8, 4.4), dpi=110)
    _A_col2 = np.array([
        [2.0,  1.0, 1.0, -1.0],
        [1.0,  2.5, -0.8, -0.5],
    ])
    _b_col2 = np.array([3.0, 0.5])

    # Shade Feasible Cone(A1, A3) vs Infeasible Cone(A1, A2)
    _cone_13 = np.array([[0.0, 0.0], _A_col2[:, 0] * 2.2, (_A_col2[:, 0] + _A_col2[:, 2]) * 1.8, _A_col2[:, 2] * 2.2])
    ax3.fill(_cone_13[:, 0], _cone_13[:, 1], color="#dcfce7", alpha=0.6, label=r"Feasible Cone $\{A_1, A_3\}$ ($b$ inside $\Rightarrow x_B \geq 0$)")
    _cone_12 = np.array([[0.0, 0.0], _A_col2[:, 0] * 2.2, (_A_col2[:, 0] + _A_col2[:, 1]) * 1.4, _A_col2[:, 1] * 1.5])
    ax3.fill(_cone_12[:, 0], _cone_12[:, 1], color="#fee2e2", alpha=0.45, label=r"Infeasible Cone $\{A_1, A_2\}$ ($b$ outside $\Rightarrow x_2 < 0$)")

    _cols_colors = ["#1d4ed8", "#7c3aed", "#15803d", "#64748b"]
    _cols_labels = ["$A_1$", "$A_2$", "$A_3$", "$A_4 = -0.5 A_1$ (dependent)"]
    for _j in range(4):
        ax3.annotate("", xy=_A_col2[:, _j], xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", lw=2.3, color=_cols_colors[_j]))
        ax3.text(_A_col2[0, _j] * 1.08, _A_col2[1, _j] * 1.08, _cols_labels[_j], fontsize=9, weight="bold", color=_cols_colors[_j])

    ax3.annotate("", xy=_b_col2, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", lw=2.8, color="#dc2626"))
    ax3.text(_b_col2[0] + 0.1, _b_col2[1], "$b = A_1 x_1 + A_3 x_3$", fontsize=10, weight="bold", color="#dc2626")

    ax3.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax3.axvline(0, color="black", lw=0.8, alpha=0.4)
    ax3.set_xlim(-1.8, 4.6)
    ax3.set_ylim(-2.0, 3.8)
    ax3.set_xlabel("Requirement Row 1")
    ax3.set_ylabel("Requirement Row 2")
    ax3.set_title("Column Geometry: Synthesizing $b$ from $m=2$ Columns")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    _df_ex21 = pl.DataFrame({
        "Variable": [f"x_{i+1}" for i in range(7)],
        "Role in Selected Set": ["Basic (in B)" if i in sel_cols else "Nonbasic (= 0)" for i in range(7)],
        "Selected Basis Value x": np.round(x_full, 3),
        "Optimal x*": np.round(opt_x_ex21, 3),
        "Optimal Basis": ["Basic (0)" if vb == 0 else "Nonbasic LB (-1)" for vb in opt_vbasis],
    })

    mo.hstack([
        mo.vstack([
            mo.md(
                f"**Selected Columns:** $\\{{ {', '.join(f'A_{i+1}' for i in sel_cols)} \\}}$ "
                f"| $\\text{{rank}}(B) = {rank_B}$, $\\det(B) = {det_B:.1f}$\n\n"
                f"{status_str}\n\n"
                f"**Optimal Basis:** Columns ${opt_basis_ex21}$ with $x^* = {np.round(opt_x_ex21, 2).tolist()}$"
            ),
            _df_ex21,
        ]),
        fig3,
    ], widths=[0.52, 0.48])
    return A_ex21, b_ex21


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Degeneracy
    """)
    return


@app.cell(hide_code=True)
def sec4_theory(mo):
    mo.md(r"""
    ### 4.1 Formal Theory: Overdetermined Corners & Colliding Bases

    At any basic solution in $\mathbb{R}^n$, we are required to have **$n$ linearly independent active constraints**. What happens if, purely by geometric coincidence or problem structure, **$n + 1$ or more constraints** pass through that exact same corner point?

    #### **Definition of Degeneracy in General Polyhedra**
    A basic solution $x \in \mathbb{R}^n$ is said to be **degenerate** if **more than $n$** of the constraints defining the polyhedron are active at $x$:

    $$
    \lvert I(x) \rvert = \lvert \{i \mid a_i'x = b_i\} \rvert > n
    $$

    **Geometric Picture:**

    * In **2D ($n = 2$):** Two lines intersect at a normal (nondegenerate) corner. If **three or more lines** intersect at the exact same point, that basic solution is **degenerate**.
    * In **3D ($n = 3$):** Three planes meet at the corner of a cube (nondegenerate). At the top apex of a 4-sided pyramid, **four planes** meet at a single point—making the apex a **degenerate** basic feasible solution.

    #### **Definition of Degeneracy in Standard Form Polyhedra**

    Consider the standard form polyhedron $P = \{x \in \mathbb{R}^n \mid Ax = b,\; x \ge 0\}$, where $A \in \mathbb{R}^{m \times n}$ has $\operatorname{rank}(A) = m$. Let $x \in \mathbb{R}^n$ be a basic solution. Then $x$ is **degenerate** if and only if **more than $n - m$ of the components of $x$ are zero**:

    $$
    \#\{j \in \{1, \dots, n\} \mid x_j = 0\} > n - m \quad \Longleftrightarrow \quad \text{at least one basic variable } x_{B(i)} = (B^{-1}b)_i = 0
    $$

    **Why Both Definitions Are Identical:** In standard form, the $m$ equality constraints $Ax = b$ are always active at $x$. The only other constraints are the $n$ nonnegativity bounds $x_j \ge 0$, which are active whenever $x_j = 0$. Therefore:

    $$
    \text{Total Active Constraints at } x = m + \#\{j \mid x_j = 0\}
    $$

    Having more than $n$ active constraints ($m + \#\{j \mid x_j = 0\} > n$) is literally the same statement as $\#\{j \mid x_j = 0\} > n - m$! And since there are $n - m$ nonbasic variables (all equal to $0$), having more than $n - m$ zeros means **at least one of the $m$ basic variables in $x_B = B^{-1}b$ must also equal $0$**.

    #### **Key Structural Consequences of Degeneracy**

    **1. Many Bases $\longrightarrow$ One Geometric Corner:** When $x$ is a *nondegenerate* BFS, exactly $m$ variables are strictly positive ($x_j > 0$), so there is **only one possible basis** $B$ for $x$. When $x$ is *degenerate*, fewer than $m$ variables are strictly positive (say $k < m$ variables are positive, and $n - k > n - m$ variables are zero). Any way of choosing $m - k$ zero variables whose columns extend the $k$ positive columns to a full-rank $m \times m$ matrix $B$ defines a **different valid basis for the exact same point $x$**!

    **2. Algorithmic Impact on Simplex:** When the Simplex method pivots at a degenerate BFS where some $x_{B(i)} = 0$, it may swap a zero basic variable out and a nonbasic variable in at level $0$—changing the basis matrix $B$ while staying at the **exact same geometric point $x$** (a step of length $\theta = 0$). Without anti-cycling rules (such as Bland's Rule or perturbation/lexicographic rules), Simplex can theoretically cycle through multiple bases sitting at the same degenerate corner.

    **3. Perturbation Splits Degeneracy:** If we slightly perturb the right-hand side $b \gets b + \epsilon$, the extra hyperplanes that passed through the exact same point separate slightly, splitting the single degenerate corner into a cluster of distinct, closely spaced **nondegenerate** corners!

    **4. Representation Dependence:** Unlike being an extreme point, **degeneracy is not a purely geometric property**—it depends on how the polyhedron is algebraically represented (for example, adding a redundant inequality that touches an existing corner turns a nondegenerate corner into a degenerate one). However, within *standard form* representations of the same polyhedron, degeneracy of a BFS is invariant.

    ---

    ### 4.2 Plain-English Deconstruction & Notation Lexicon

    | Concept | Nondegenerate Basic Feasible Solution | Degenerate Basic Feasible Solution |
    | :--- | :--- | :--- |
    | **Active Constraints in $\mathbb{R}^n$** | Exactly $n$ constraints are active ($\lvert I(x) \rvert = n$). | Strictly more than $n$ constraints are active ($\lvert I(x) \rvert > n$). |
    | **Zero Variables in Standard Form** | Exactly $n - m$ variables are zero (only the nonbasic variables $x_N = 0$). | More than $n - m$ variables are zero (all $x_N = 0$ **plus** at least one basic variable $x_{B(i)} = 0$). |
    | **Basic Variable Values $x_B = B^{-1}b$** | Strictly positive: $x_B > 0$ ($x_{B(i)} > 0$ for all $i = 1, \dots, m$). | Nonnegative with zeros: $x_B \ge 0$, with $x_{B(i)} = 0$ for at least one $i$. |
    | **Basis $\leftrightarrow$ Extreme Point Map** | **Unique (1-to-1):** Only one basis $B$ produces $x$. | **Many-to-1:** Multiple distinct bases $B_1, B_2, \dots$ can produce the exact same $x$. |

    ---

    ### 4.3 Worked Analytical Example

    Return to the 3D polyhedron ($n = 3$) from Section 3.4:

    $$
    \begin{aligned}
    \text{(1)}\quad x_1 + x_2 + 2x_3 &\le 8 \\
    \text{(2)}\quad x_2 + 6x_3 &\le 12 \\
    \text{(3)}\quad x_1 &\le 4 \\
    \text{(4)}\quad x_2 &\le 6 \\
    \text{(5, 6, 7)}\quad x_1, x_2, x_3 &\ge 0
    \end{aligned}
    $$

    In standard form $Ax = b, x \ge 0$ (adding slacks $x_4, x_5, x_6, x_7$), we have $n = 7$, $m = 4$, and $n - m = 3$.

    **1. Nondegenerate BFS $x^{(A)} = (2, 6, 0)'$ in $\mathbb{R}^3$:**

    * Plug into the 7 constraints: (1) $2+6+0 = 8$ **(active)**; (2) $6 + 0 = 6 < 12$; (3) $2 < 4$; (4) $6 = 6$ **(active)**; (5) $x_1=2>0$; (6) $x_2=6>0$; (7) $x_3=0$ **(active)**.
    * Exactly $3 = n$ constraints are active! In $\mathbb{R}^7$, $x^{(A)} = (2, 6, 0, 0, 6, 2, 0)'$ has **3 zeros** ($= n-m$), and its **only** basis is $\{1, 2, 5, 6\}$.

    **2. Degenerate BFS $x^{(B)} = (4, 0, 2)'$ in $\mathbb{R}^3$:**

    * Constraint checks: (1) $4 + 0 + 2(2) = 8$ **(active, $x_4 = 0$)**; (2) $0 + 6(2) = 12$ **(active, $x_5 = 0$)**; (3) $x_1 = 4$ **(active, $x_6 = 0$)**; (4) $x_2 = 0 < 6$ *(inactive, $x_7 = 6 > 0$)*; (6) $x_2 = 0$ **(active!)**.
    * **Four constraints** are active at $(4, 0, 2)'$ in $\mathbb{R}^3$! In standard form $\mathbb{R}^7$, $x^{(B)} = (4, 0, 2, 0, 0, 0, 6)'$ has **$4 > 3$ zeros** ($x_2 = x_4 = x_5 = x_6 = 0$).
    * Every valid basis for $x^{(B)}$ must include the 3 positive columns $\{1, 3, 7\}$ ($x_1=4, x_3=2, x_7=6$) plus **any 1** of the 4 zero columns $\{2, 4, 5, 6\}$ that keeps $B$ invertible! As verified below, **all 4 bases** $\{1, 2, 3, 7\}$, $\{1, 3, 4, 7\}$, $\{1, 3, 5, 7\}$, and $\{1, 3, 6, 7\}$ are nonsingular and yield the **exact same BFS** $x^{(B)} = (4, 0, 2, 0, 0, 0, 6)'$!
    """)
    return


@app.cell
def sec4_ui(mo):
    eps_degen_slider = mo.ui.slider(
        start=-0.6, stop=0.6, step=0.1, value=0.0,
        label=r"Perturbation $\epsilon$ on 2D line 3 ($x_1 + 2x_2 \leq 4 + \epsilon$) to split a degenerate 2D corner:"
    )
    eps_degen_slider
    return (eps_degen_slider,)


@app.cell
def sec4_code(A_ex21, b_ex21, eps_degen_slider, linprog, mo, np, pl, plt):
    # Part A: Verify all bases that collide at the Degenerate BFS x = (4, 0, 2, 0, 0, 0, 6)
    from itertools import combinations as _comb4

    target_degen_x = np.array([4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 6.0])
    colliding_bases_records = []

    for _b_cols in _comb4(range(7), 4):
        _B_mat = A_ex21[:, list(_b_cols)]
        if np.linalg.matrix_rank(_B_mat) == 4:
            _xb = np.linalg.solve(_B_mat, b_ex21)
            _x_cand = np.zeros(7)
            _x_cand[list(_b_cols)] = _xb
            if np.allclose(_x_cand, target_degen_x):
                _zero_basic = [f"x_{i+1}=0" for i, val in zip(_b_cols, _xb) if abs(val) < 1e-9]
                colliding_bases_records.append({
                    "Basis Indices B": f"{{{', '.join(str(i+1) for i in _b_cols)}}}",
                    "det(B)": float(np.round(np.linalg.det(_B_mat), 2)),
                    "Basic Variable Values x_B = B^-1 b": str(np.round(_xb, 1).tolist()),
                    "Zero Basic Variable(s)": ", ".join(_zero_basic),
                    "Resulting x in R^7": str(np.round(_x_cand, 1).tolist()),
                })

    # Part B: 2D Visual Demonstration of Degeneracy & Perturbation + Optimal Solution
    # Three constraints in R^2 meeting at the exact same corner (2, 1) when eps = 0:
    # (1) x1 + x2 <= 3
    # (2) x1 <= 2
    # (3) x1 + 2*x2 <= 4 + eps
    # (4, 5) x1 >= 0, x2 >= 0
    _eps = eps_degen_slider.value
    _A_2d_deg = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 2.0],
    ])
    _b_2d_deg = np.array([3.0, 2.0, 4.0 + _eps])

    _res4 = linprog(np.array([-2.0, -3.0]), A_ub=_A_2d_deg, b_ub=_b_2d_deg, bounds=(0.0, None), method="highs")
    _opt_x4 = _res4.x.copy()
    _slacks4 = _b_2d_deg - (_A_2d_deg @ _opt_x4)
    _active_count = int(np.sum(np.abs(_slacks4) < 1e-7) + np.sum(np.abs(_opt_x4) < 1e-7))

    fig4, ax4 = plt.subplots(figsize=(7.2, 4.5), dpi=110)
    _x1v = np.linspace(0.0, 3.6, 400)
    ax4.plot(_x1v, 3.0 - _x1v, lw=2.2, color="#2563eb", label=r"(1) $x_1 + x_2 = 3$")
    ax4.axvline(2.0, lw=2.2, color="#059669", label=r"(2) $x_1 = 2$")
    ax4.plot(_x1v, (4.0 + _eps - _x1v) / 2.0, lw=2.2, ls="--", color="#dc2626",
             label=rf"(3) $x_1 + 2x_2 = {4.0 + _eps:.1f}$ ($\epsilon = {_eps:+.1f}$)")

    # Compute intersections of (1)&(2), (1)&(3), (2)&(3)
    _p12 = np.array([2.0, 1.0])
    _p13 = np.array([2.0 - _eps, 1.0 + _eps])
    _p23 = np.array([2.0, (2.0 + _eps) / 2.0])

    for _pt, _lbl, _col in [(_p12, "{1,2}", "#1e40af"), (_p13, "{1,3}", "#991b1b"), (_p23, "{2,3}", "#065f46")]:
        ax4.scatter(*_pt, s=85, color=_col, edgecolor="black", zorder=6)
        ax4.text(_pt[0] + 0.06, _pt[1] + 0.06, f"{_lbl}: ({_pt[0]:.2f},{_pt[1]:.2f})", fontsize=8.5, color=_col, weight="bold")

    ax4.scatter(*_opt_x4, s=160, marker="*", color="#f59e0b", edgecolor="black", zorder=8,
                label=f"Optimal Corner: ({_opt_x4[0]:.2f}, {_opt_x4[1]:.2f})")

    ax4.set_xlim(0.0, 3.5)
    ax4.set_ylim(0.0, 2.6)
    ax4.set_xlabel("$x_1$")
    ax4.set_ylabel("$x_2$")
    ax4.set_title(
        "DEGENERATE Corner at (2, 1): 3 Lines Meet (|I(x)| = 3 > n = 2)"
        if abs(_eps) < 1e-9
        else f"PERTURBED (ε = {_eps:+.1f}): Degenerate Corner Splits into Nondegenerate Corners!"
    )
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.hstack([
        mo.vstack([
            mo.md(
                f"### Standard-Form Degeneracy ($n=7, m=4$): **4 Distinct Bases** Colliding at $x = (4, 0, 2, 0, 0, 0, 6)'$\n"
                f"Because $x$ has $4 > n-m=3$ zeros, choosing any 1 of the 4 zero variables ($x_2, x_4, x_5, x_6$) to join the 3 positive variables ($x_1=4, x_3=2, x_7=6$) forms a valid basis for the **exact same geometric corner**:"
            ),
            pl.DataFrame(colliding_bases_records),
            mo.md(
                f"**2D Perturbation Status ($\\epsilon = {_eps:+.1f}$):** At the optimal corner $x^* = ({_opt_x4[0]:.2f}, {_opt_x4[1]:.2f})'$, "
                f"**{_active_count}** constraints are active in $\\mathbb{{R}}^2$ "
                f"({'**DEGENERATE** ($>2$)' if _active_count > 2 else '**NONDEGENERATE** ($=2$)'})."
            ),
        ]),
        fig4,
    ], widths=[0.54, 0.46])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Existence of Extreme Points
    """)
    return


@app.cell(hide_code=True)
def sec5_theory(mo):
    mo.md(r"""
    ### 5.1 Formal Theory: Lines vs. Corners & The Rank Condition

    Does every nonempty polyhedron have at least one extreme point? **No!**
    For example, in $\mathbb{R}^2$, the infinite horizontal strip $P_{\text{strip}} = \{(x_1, x_2) \in \mathbb{R}^2 \mid 0 \le x_2 \le 2\}$ is a perfectly valid polyhedron, yet it has **zero extreme points** (every point is the midpoint of a horizontal segment $(x_1 - \epsilon, x_2)$ and $(x_1 + \epsilon, x_2)$). Why? Because $P_{\text{strip}}$ contains an **infinite line**!

    #### **Definition 2.12 (Polyhedron Containing a Line)**
    A polyhedron $P \subseteq \mathbb{R}^n$ **contains a line** if there exists a point $x \in P$ and a **nonzero** direction vector $d \in \mathbb{R}^n$ ($d \ne 0$) such that:

    $$
    x + \lambda d \in P \quad \text{for ALL scalars } \lambda \in (-\infty, +\infty)
    $$

    > **Warning — Line vs. Half-Line (Ray):** Do not confuse containing a *two-way infinite line* ($\forall \lambda \in \mathbb{R}$) with being *unbounded* (containing a one-way ray $\forall \lambda \ge 0$). For instance, the positive quadrant $\{(x_1, x_2) \mid x_1 \ge 0, x_2 \ge 0\}$ is unbounded and contains rays, but it **does not contain any full line**, and it **does** have an extreme point at $(0, 0)'$!

    #### **Theorem 2.6 (Necessary and Sufficient Conditions for Existence of Extreme Points)**
    Suppose the polyhedron $P = \{x \in \mathbb{R}^n \mid a_i'x \ge b_i,\; i = 1, \dots, m\}$ is nonempty. Then the following three statements are equivalent:

    $$
    \boxed{\text{(a) } P \text{ has at least one extreme point} \;\Longleftrightarrow\; \text{(b) } P \text{ does NOT contain a line} \;\Longleftrightarrow\; \text{(c) } \operatorname{rank}(\{a_1, \dots, a_m\}) = n}
    $$

    *(where $\operatorname{rank}(\{a_1, \dots, a_m\}) = n$ means there exist $n$ linearly independent vectors among the constraint normals $a_1, \dots, a_m$, i.e., $\operatorname{rank}(A) = n$).*

    #### **Constructive Proof of Theorem 2.6 *(The "Slide-Until-You-Hit-a-Wall" Procedure)***

    **1. $\text{(b) No Line} \implies \text{(a) Extreme Point Exists}$:**

    Start at any feasible point $x \in P$ with active set $I = \{i \mid a_i'x = b_i\}$. Let $k = \operatorname{rank}(\{a_i \mid i \in I\})$ be the number of linearly independent active constraints at $x$.

    * If $k = n$, then $x$ is already a Basic Feasible Solution (and by Theorem 2.3, an extreme point)!
    * If $k < n$, the active normals $\{a_i \mid i \in I\}$ span a subspace of dimension $k < n$, so there exists a nonzero direction $d \ne 0$ orthogonal to all currently active normals:

    $$
    a_i'd = 0 \quad \text{for all } i \in I
    $$

    Consider the line $y(\lambda) = x + \lambda d$ for $\lambda \in \mathbb{R}$. Notice that for every active constraint $i \in I$:

    $$
    a_i'y(\lambda) = a_i'x + \lambda(a_i'd) = b_i + 0 = b_i \quad (\forall \lambda \in \mathbb{R})
    $$

    So sliding along $d$ **never leaves any currently active wall**! Meanwhile, because $P$ contains no line, sliding along $+\lambda$ or $-\lambda$ cannot stay inside $P$ forever—it must eventually hit some previously inactive constraint $j \notin I$ at some threshold $\lambda^*$:

    $$
    a_j'(x + \lambda^* d) = b_j \quad \text{while } x + \lambda^* d \in P
    $$

    **Why does hitting wall $j$ strictly increase the rank of active constraints to $k + 1$?** Because $j \notin I$ at $x$, we had $a_j'x > b_j$, whereas at $x + \lambda^* d$ we have $a_j'x + \lambda^*(a_j'd) = b_j$. Subtracting gives $\lambda^*(a_j'd) = b_j - a_j'x \ne 0 \implies \mathbf{a_j'd \ne 0}$. Since $d$ is orthogonal to every linear combination of $\{a_i \mid i \in I\}$ ($a_i'd = 0$) but is **not** orthogonal to $a_j$ ($a_j'd \ne 0$), $a_j$ **cannot be a linear combination of $\{a_i \mid i \in I\}$**! Thus the new point $x_{\text{new}} = x + \lambda^* d \in P$ has at least $k + 1$ linearly independent active constraints. Repeating at most $n - k$ steps reaches $n$ linearly independent active constraints—an extreme point!

    **2. $\text{(a) Extreme Point Exists} \implies \text{(c) } \operatorname{rank}(A) = n$:**

    By Theorem 2.3, an extreme point is a BFS, which requires $n$ linearly independent active normals in $\{a_1, \dots, a_m\}$.

    **3. $\text{(c) } \operatorname{rank}(A) = n \implies \text{(b) No Line}$:**

    Suppose $P$ contained a line $x + \lambda d$ ($\forall \lambda \in \mathbb{R}$) with $d \ne 0$. Then for every constraint $i = 1, \dots, m$:

    $$
    a_i'x + \lambda(a_i'd) \ge b_i \quad \text{for all } \lambda \in (-\infty, +\infty)
    $$

    If $a_i'd \ne 0$, sending $\lambda \to -\operatorname{sign}(a_i'd)\cdot \infty$ makes the left-hand side $-\infty < b_i$, violating feasibility! Thus we **must** have $a_i'd = 0$ for all $i = 1, \dots, m$ (i.e., $Ad = 0$). But since $A$ has $n$ linearly independent rows ($\operatorname{rank}(A) = n$), $Ad = 0 \implies d = 0$, a contradiction! $\blacksquare$

    #### **Corollary 2.2 (Guaranteed Extreme Points for Bounded & Standard Form Polyhedra)**
    **Every nonempty bounded polyhedron** and **every nonempty polyhedron in standard form** $P = \{x \in \mathbb{R}^n \mid Ax = b,\; x \ge 0\}$ has **at least one extreme point (BFS)**.

    * *Why?* A bounded polyhedron obviously cannot contain an infinite line. And a standard-form polyhedron is a subset of the nonnegative orthant $\{x \in \mathbb{R}^n \mid I_n x \ge 0\}$, whose $n$ normal vectors are the standard basis vectors $e_1, \dots, e_n$ ($\operatorname{rank}(I_n) = n$), so no line $x + \lambda d$ ($\lambda \in \mathbb{R}$) can stay nonnegative for both $\lambda \to +\infty$ and $\lambda \to -\infty$!

    ---

    ### 5.2 Plain-English Deconstruction & Notation Lexicon

    | Mathematical Condition | Plain-English Translation | Geometric Meaning |
    | :--- | :--- | :--- |
    | $x + \lambda d \in P \;\; (\forall \lambda \in \mathbb{R})$ | "Line $x + \lambda d$ lies entirely inside $P$ for both $+\infty$ and $-\infty$" | You can walk forever forward **and** forever backward along direction $d$ without ever hitting a constraint wall. |
    | $Ad = 0 \;\; (d \ne 0)$ | "$d$ lies in the nullspace $\mathcal{N}(A)$ of the constraint matrix $A$" | Moving along $d$ produces zero change in **every** constraint ($a_i'(x+\lambda d) = a_i'x$), so the polyhedron is an infinite cylinder/trough along $d$. |
    | $\operatorname{rank}(A) = n$ | "The $m \times n$ matrix $A$ has full column rank $n$" | The nullspace $\mathcal{N}(A) = \{0\}$ is trivial, making it impossible to have an infinite line $Ad = 0$ inside $P$. |

    ---

    ### 5.3 Worked Analytical Example *(Constructive Path to an Extreme Point)*

    Consider $P = \{x \in \mathbb{R}^2 \mid x_1 \ge 0,\; x_2 \ge 0,\; x_1 + 2x_2 \ge 2,\; -x_1 + x_2 \ge -3\}$.
    Start at the strictly interior point $x^{(0)} = (2, 2)' \in P$:

    1. **At $x^{(0)} = (2, 2)'$ ($k = 0$ active constraints):** $I(x^{(0)}) = \emptyset$. Pick any nonzero direction, say $d^{(0)} = (-1, 0)'$. Move along $x^{(0)} + \lambda d^{(0)} = (2 - \lambda, 2)'$ for $\lambda > 0$: it hits $x_1 = 0$ (Constraint 1) at $\lambda^* = 2$, reaching $x^{(1)} = (0, 2)'$.
    2. **At $x^{(1)} = (0, 2)'$ ($k = 1$ active constraint, $I = \{1\}$ with $a_1 = (1, 0)'$):** Choose $d^{(1)} \ne 0$ orthogonal to $a_1 = (1, 0)'$: $d^{(1)} = (0, -1)'$ ($a_1'd^{(1)} = 0$). Move along $x^{(1)} + \lambda d^{(1)} = (0, 2 - \lambda)'$ for $\lambda > 0$: Constraint 1 stays active ($x_1 = 0$), and at $\lambda^* = 1$ we hit Constraint 3 ($x_1 + 2x_2 = 0 + 2(1) = 2$) at $x^{(2)} = (0, 1)'$.
    3. **At $x^{(2)} = (0, 1)'$ ($k = 2 = n$ active constraints, $I = \{1, 3\}$):** Normals $a_1 = (1, 0)'$ and $a_3 = (1, 2)'$ are linearly independent ($\operatorname{rank} = 2 = n$). We have constructively reached the **Extreme Point / BFS $x^{(2)} = (0, 1)'$**!
    """)
    return


@app.cell
def sec5_code(mo, np, pl, plt):
    # Interactive Theorem 2.6 Constructive "Slide-to-Corner" Algorithm & Line-vs-Corner Comparison
    A_sec5 = np.array([
        [ 1.0,  0.0],   # 1: x1 >= 0
        [ 0.0,  1.0],   # 2: x2 >= 0
        [ 1.0,  2.0],   # 3: x1 + 2*x2 >= 2
        [-1.0,  1.0],   # 4: -x1 + x2 >= -3  (x1 - x2 <= 3)
    ])
    b_sec5 = np.array([0.0, 0.0, 2.0, -3.0])

    def constructive_find_extreme_point(A, b, x_start):
        """Exact implementation of Theorem 2.6 constructive proof (Slide-Until-You-Hit-a-Wall)."""
        x_curr = x_start.astype(float).copy()
        n_dim = A.shape[1]
        path = [x_curr.copy()]
        step_logs = []

        for step in range(n_dim + 1):
            slacks = A @ x_curr - b
            active_idx = [i for i, s in enumerate(slacks) if abs(s) < 1e-8]
            A_act = A[active_idx, :] if active_idx else np.zeros((0, n_dim))
            rk = int(np.linalg.matrix_rank(A_act)) if len(active_idx) > 0 else 0
            step_logs.append({
                "Step": step,
                "Point x^(k)": f"({x_curr[0]:.2f}, {x_curr[1]:.2f})",
                "Active Constraints I(x)": str([i + 1 for i in active_idx]),
                "Active Rank k": rk,
                "Status": "Extreme Point / BFS Reached (k = n)!" if rk == n_dim else f"Sliding in nullspace (k = {rk} < {n_dim})",
            })
            if rk == n_dim:
                break
            # Find nonzero direction d in nullspace of A_act (a_i' d = 0 for all i in I)
            if rk == 0:
                d = np.array([-1.0, -0.5])
            else:
                _, _, vh = np.linalg.svd(A_act)
                d = vh[-1, :]  # Last row of V^T is orthogonal to row space of A_act
            # Find smallest positive lambda* (or negative lambda*) that hits an inactive wall a_j'(x + lam*d) = b_j
            best_lam = None
            for sign in [1.0, -1.0]:
                d_try = sign * d
                rates = A @ d_try
                cands = []
                for j in range(len(b)):
                    if j not in active_idx and rates[j] < -1e-9:
                        cands.append(-slacks[j] / rates[j])
                if cands:
                    best_lam = min(cands)
                    d = d_try
                    break
            x_curr = x_curr + best_lam * d
            path.append(x_curr.copy())
        return np.array(path), step_logs

    _path_sec5, _logs_sec5 = constructive_find_extreme_point(A_sec5, b_sec5, np.array([2.5, 2.2]))

    fig5, (ax5_left, ax5_right) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=110)

    # Left panel: Line-free polyhedron (rank(A) = 2) + Constructive Path
    _x5g = np.linspace(-0.3, 4.8, 300)
    ax5_left.axvline(0.0, color="#64748b", lw=1.6, label="1: $x_1 = 0$")
    ax5_left.axhline(0.0, color="#64748b", lw=1.6, ls="--", label="2: $x_2 = 0$")
    ax5_left.plot(_x5g, (2.0 - _x5g) / 2.0, color="#2563eb", lw=1.8, label="3: $x_1 + 2x_2 = 2$")
    ax5_left.plot(_x5g, _x5g - 3.0, color="#7c3aed", lw=1.8, label="4: $-x_1 + x_2 = -3$")

    _poly5 = np.array([[0.0, 4.2], [0.0, 1.0], [2.0, 0.0], [3.0, 0.0], [4.8, 1.8], [4.8, 4.2]])
    ax5_left.fill(_poly5[:, 0], _poly5[:, 1], color="#dbeafe", alpha=0.5)

    # Plot constructive path x^(0) -> x^(1) -> x^(2)
    for _k in range(len(_path_sec5) - 1):
        ax5_left.annotate(
            "", xy=_path_sec5[_k + 1], xytext=_path_sec5[_k],
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#dc2626")
        )
    for _k, _pt in enumerate(_path_sec5):
        ax5_left.scatter(*_pt, s=90, color="#dc2626" if _k < len(_path_sec5)-1 else "#16a34a", edgecolor="black", zorder=6)
        ax5_left.text(_pt[0] + 0.1, _pt[1] + 0.12, f"$x^{{({_k})}}$ (rank {_k})", fontsize=8.5, weight="bold")

    ax5_left.set_xlim(-0.4, 4.6)
    ax5_left.set_ylim(-0.4, 3.8)
    ax5_left.set_xlabel("$x_1$")
    ax5_left.set_ylabel("$x_2$")
    ax5_left.set_title("Theorem 2.6: $\\mathrm{rank}(A)=2=n$\nNo Line $\\Rightarrow$ Constructive Slide Finds BFS $x^{(2)}$")
    ax5_left.legend(loc="upper right", fontsize=7.5)
    ax5_left.grid(True, alpha=0.3)

    # Right panel: Polyhedron containing a line (rank(A) = 1 < n = 2) -> NO Extreme Points!
    # P_line = { (x1, x2) | 1 <= x1 - x2 <= 3 } -> normals (1, -1) and (-1, 1) have rank 1!
    ax5_right.plot(_x5g, _x5g - 1.0, color="#2563eb", lw=2, label=r"$x_1 - x_2 \geq 1$")
    ax5_right.plot(_x5g, _x5g - 3.0, color="#7c3aed", lw=2, label=r"$-x_1 + x_2 \geq -3$")
    ax5_right.fill_between(_x5g, _x5g - 3.0, _x5g - 1.0, color="#fef3c7", alpha=0.7, label="Infinite Strip $P_{\\mathrm{line}}$")
    ax5_right.plot(_x5g, _x5g - 2.0, "r--", lw=2.2, label=r"Infinite Line $x + \lambda(1,1)' \in P\;\;(\forall \lambda \in \mathbb{R})$")
    ax5_right.set_xlim(-0.4, 4.6)
    ax5_right.set_ylim(-2.5, 3.5)
    ax5_right.set_xlabel("$x_1$")
    ax5_right.set_ylabel("$x_2$")
    ax5_right.set_title("Rank-Deficient $\\mathrm{rank}(A)=1 < n=2$:\nContains Line $d=(1,1)' \\Rightarrow$ ZERO Extreme Points!")
    ax5_right.legend(loc="upper left", fontsize=7.5)
    ax5_right.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        fig5,
        pl.DataFrame(_logs_sec5),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Optimality of Extreme Points (The Fundamental Theorem of LP)
    """)
    return


@app.cell(hide_code=True)
def sec6_theory(mo):
    mo.md(r"""
    ### 6.1 Formal Theory: Why We Only Need to Search Corner Points

    We now arrive at the crowning geometric result of Linear Programming: even though a polyhedron $P$ contains an uncountably infinite continuum of points, **if $P$ has at least one extreme point and the optimal cost is finite, the global optimum is guaranteed to be achieved at an extreme point (BFS)**.

    #### **Theorem 2.7 (Optimality of Extreme Points)**
    Consider the linear programming problem of minimizing $c'x$ over a polyhedron $P = \{x \in \mathbb{R}^n \mid Ax \ge b\}$. Suppose that $P$ has at least one extreme point and that there exists an optimal solution. Then **there exists an optimal solution which is an extreme point of $P$**.

    **Proof *(The "Optimal Face $Q$" Argument)*:**

    1. Let $v = \min_{x \in P} c'x$ be the optimal objective value, and define the **set of all optimal solutions** (the *optimal face*):

    $$
    Q = \{x \in \mathbb{R}^n \mid Ax \ge b,\; c'x = v\}
    $$

    2. By assumption, $Q$ is nonempty. Because $Q$ is defined by linear constraints, $Q$ is itself a **polyhedron**! Furthermore, since $Q \subseteq P$ and $P$ contains no lines (Theorem 2.6), $Q$ cannot contain a line either. Therefore, by Theorem 2.6, **$Q$ has at least one extreme point $x^*$**.
    3. **We claim that $x^*$ (an extreme point of $Q$) is ALSO an extreme point of $P$:** Suppose, for contradiction, that $x^*$ were *not* an extreme point of $P$. Then there exist $y, z \in P$ ($y \ne x^*, z \ne x^*$) and $\lambda \in (0, 1)$ such that:

    $$
    x^* = \lambda y + (1 - \lambda)z
    $$

    Evaluate the linear objective $c'(\cdot)$ on both sides:

    $$
    v = c'x^* = \lambda c'y + (1 - \lambda)c'z
    $$

    Since $y, z \in P$ are feasible and $v$ is the minimum cost over all of $P$, we have **$c'y \ge v$** and **$c'z \ge v$**. Because $\lambda \in (0, 1)$ is strictly between $0$ and $1$, the weighted average $\lambda c'y + (1-\lambda)c'z$ can equal $v$ **if and only if both $c'y = v$ and $c'z = v$**! But $c'y = v$ and $c'z = v$ mean that **$y \in Q$ and $z \in Q$**, which contradicts the fact that $x^*$ is an extreme point of $Q$! Therefore, $x^*$ must be an extreme point of $P$, and since $x^* \in Q$, $x^*$ is an optimal extreme point of $P$. $\blacksquare$

    ---

    #### **Theorem 2.8 (Finite Optimal Cost Guarantees an Optimal Extreme Point)**
    Consider minimizing $c'x$ over a polyhedron $P$ that has at least one extreme point. Then **either the optimal cost is equal to $-\infty$, or there exists an extreme point which is optimal**.

    **Proof *(Cost-Nonincreasing Rank-Ascent)*:**

    Assume the optimal cost is finite ($> -\infty$). Take any feasible point $x \in P$ with $k < n$ linearly independent active constraints ($I = \{i \mid a_i'x = b_i\}$). We construct a new feasible point $y \in P$ with **strictly higher active rank ($\ge k + 1$)** and **no worse cost ($c'y \le c'x$)**:

    1. Since $k < n$, choose a nonzero vector $d \ne 0$ orthogonal to all active normals ($a_i'd = 0$ for all $i \in I$). By replacing $d$ with $-d$ if necessary, ensure that **$c'd \le 0$**.
    2. **Case 1 ($c'd < 0$):** Move along the ray $y(\lambda) = x + \lambda d$ for $\lambda > 0$. If the entire half-line stayed in $P$, then $c'y(\lambda) = c'x + \lambda(c'd) \to -\infty$ as $\lambda \to \infty$, contradicting that the optimal cost is finite! Thus the ray must exit $P$ at some $\lambda^* > 0$, hitting a new linearly independent constraint $j \notin I$ at $y = x + \lambda^* d$ with $c'y < c'x$ and active rank $\ge k + 1$.
    3. **Case 2 ($c'd = 0$):** Move along the line $y(\lambda) = x + \lambda d$ ($\lambda \in \mathbb{R}$). Since $P$ contains no line, it hits a new linearly independent constraint $j \notin I$ at some $\lambda^*$, giving $y = x + \lambda^* d$ with $c'y = c'x$ and active rank $\ge k + 1$.
    4. Repeating at most $n - k$ steps produces a Basic Feasible Solution (extreme point) $w \in P$ with $c'w \le c'x$.
    5. By Corollary 2.1, $P$ has only a **finite number** of extreme points $\{w^1, \dots, w^r\}$. Let $w^* \in \arg\min_{i=1,\dots,r} c'w^i$ be the best among these finitely many extreme points. Since every $x \in P$ has some extreme point $w^i$ with $c'w^i \le c'x$, we have $c'w^* \le c'w^i \le c'x$ for **all** $x \in P$! Thus $w^*$ is globally optimal. $\blacksquare$

    #### **Corollary 2.3 (No "Unattained Finite Infimum" in LP)**
    Consider minimizing $c'x$ over **any** nonempty polyhedron $P$. Then either the optimal cost is $-\infty$, or there exists an optimal solution $x^* \in P$.

    * **Contrast with Nonlinear Optimization:** In nonlinear optimization, $\min_{x \ge 1} \frac{1}{x}$ has a finite infimum of $0 > -\infty$, yet **no optimal solution exists** (since $1/x > 0$ for every finite $x \ge 1$). Theorem 2.8 and Corollary 2.3 prove that a linear program with finite infimum **always attains** its minimum!

    ---

    ### 6.2 Plain-English Deconstruction & Notation Lexicon

    | Symbol / Expression | Plain-English Meaning | Geometric Role |
    | :--- | :--- | :--- |
    | $v = \min_{x \in P} c'x$ | **Optimal Objective Value:** The lowest possible cost achievable anywhere in $P$. | Defines the level set $\{x \mid c'x = v\}$ that just kisses the boundary of $P$. |
    | $Q = \{x \in P \mid c'x = v\}$ | **Optimal Solution Set (Optimal Face):** All feasible points that tie for the best cost $v$. | Can be a single corner (0D vertex), a whole edge between two corners (1D segment), or a higher-dimensional facet of $P$. Even when $Q$ is a whole edge or facet, its corners are corners of $P$! |
    | $c'd \le 0$ with $a_i'd = 0\; (\forall i \in I)$ | **Descent/Flat Direction Along Active Walls:** Sliding along currently active walls in a direction where cost decreases or stays flat. | Guarantees we can always slide from any interior or edge point to a corner without ever increasing the objective $c'x$! |

    ---

    ### 6.3 Worked Analytical Example *(Multiple Optimal Solutions vs. Unique Extreme Point Optimum)*

    Consider the bounded polygon $P \subset \mathbb{R}^2$ with 4 extreme points: $w^1 = (0, 0)'$, $w^2 = (2, 0)'$, $w^3 = (3, 1)'$, $w^4 = (0, 4)'$.

    * **1. Unique Optimal Extreme Point:** Minimize $c'x = -x_1 - 2x_2$. Evaluate $c'w^i$ at the 4 extreme points: $c'w^1 = 0$, $c'w^2 = -2$, $c'w^3 = -3 - 2 = -5$, $c'w^4 = 0 - 8 = -8$. Unique optimal set: $Q = \{w^4\} = \{(0, 4)'\}$ with $v = -8$.
    * **2. Multiple Optimal Solutions (Optimal Edge $Q$):** Minimize $c'x = -x_1 - x_2$ (parallel to constraint $x_1 + x_2 \le 4$!). Evaluate $c'w^i$: $c'w^1 = 0$, $c'w^2 = -2$, **$c'w^3 = -4$**, **$c'w^4 = -4$**. Here the optimal set $Q = \{x \in P \mid -x_1 - x_2 = -4\} = \{\lambda(3, 1)' + (1-\lambda)(0, 4)' \mid \lambda \in [0, 1]\}$ is the **entire edge** connecting $w^3$ and $w^4$! And as Theorem 2.7 proves, the **extreme points of $Q$** are $w^3 = (3, 1)'$ and $w^4 = (0, 4)'$—which are **extreme points of $P$**!
    """)
    return


@app.cell
def sec6_ui(mo):
    theta_deg_slider = mo.ui.slider(
        start=0, stop=360, step=15, value=225,
        label=r"Cost vector angle $\theta$ (degrees) for $c(\theta) = (\cos\theta, \sin\theta)'$ in $\min_{x \in P} c(\theta)'x$ (try $225^\circ$ for an optimal edge $Q$!):"
    )
    theta_deg_slider
    return (theta_deg_slider,)


@app.cell
def sec6_code(A_sec2, b_sec2, linprog, mo, np, pl, plt, theta_deg_slider):
    # Interactive Theorem 2.7 & 2.8 Laboratory: Optimal Face Q, HiGHS Extreme Point Optimum & Cost-Nonincreasing Rank-Ascent
    _theta_rad = np.deg2rad(theta_deg_slider.value)
    c_sec6 = np.array([np.cos(_theta_rad), np.sin(_theta_rad)])

    # Solve min c'x over P (from Section 2) using SciPy HiGHS
    _res6 = linprog(c_sec6, A_ub=-A_sec2, b_ub=-b_sec2, bounds=(None, None), method="highs")
    opt_x6 = _res6.x.copy()
    v_opt6 = float(_res6.fun)

    # Evaluate c'w^i at all 4 extreme points of P
    _ext_pts6 = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [0.0, 4.0]])
    _ext_costs6 = _ext_pts6 @ c_sec6
    _opt_ext_mask = np.isclose(_ext_costs6, v_opt6, atol=1e-6)
    _opt_ext_pts = _ext_pts6[_opt_ext_mask]

    # Trace Theorem 2.8 Cost-Nonincreasing Rank-Ascent from interior point x^(0) = (1.0, 1.2)
    _x_curr6 = np.array([1.0, 1.2])
    _ascent_path6 = [_x_curr6.copy()]
    for _step in range(2):
        _slk = A_sec2 @ _x_curr6 - b_sec2
        _act = [i for i, s in enumerate(_slk) if abs(s) < 1e-7]
        if len(_act) == 0:
            # Rank 0: move in direction -c (steepest descent) until hitting a wall
            _d = -c_sec6
        else:
            # Rank 1: orthogonal to active normal a_i, chosen so c'd <= 0
            _a_act = A_sec2[_act[0]]
            _d = np.array([-_a_act[1], _a_act[0]])
            if float(c_sec6 @ _d) > 0:
                _d = -_d
        _rates = A_sec2 @ _d
        _lams = [-_slk[j] / _rates[j] for j in range(4) if j not in _act and _rates[j] < -1e-9]
        if _lams:
            _x_curr6 = _x_curr6 + min(_lams) * _d
            _ascent_path6.append(_x_curr6.copy())
    _ascent_path6 = np.array(_ascent_path6)

    fig6, ax6 = plt.subplots(figsize=(7.4, 4.8), dpi=110)
    ax6.fill(_ext_pts6[:, 0], _ext_pts6[:, 1], color="#f1f5f9", edgecolor="#334155", lw=1.8, label="Polyhedron $P$")

    # Objective level curves (isocost lines)
    _X1, _X2 = np.meshgrid(np.linspace(-0.5, 4.0, 100), np.linspace(-0.5, 4.5, 100))
    _Z = c_sec6[0] * _X1 + c_sec6[1] * _X2
    ax6.contour(_X1, _X2, _Z, levels=12, cmap="coolwarm", alpha=0.45, linestyles=":")

    # Highlight Optimal Face Q = {x in P | c'x = v*}
    if len(_opt_ext_pts) >= 2:
        ax6.plot(_opt_ext_pts[:, 0], _opt_ext_pts[:, 1], color="#16a34a", lw=6.0, zorder=6,
                 label="Optimal Face $Q$ (Entire Edge! Extreme pts of $Q \\subset$ Extreme pts of $P$)")

    # Plot Theorem 2.8 Cost-Nonincreasing Rank-Ascent path
    for _k in range(len(_ascent_path6) - 1):
        ax6.annotate("", xy=_ascent_path6[_k + 1], xytext=_ascent_path6[_k],
                     arrowprops=dict(arrowstyle="->", lw=2.4, color="#7c3aed"))
    ax6.scatter(_ascent_path6[:, 0], _ascent_path6[:, 1], color="#7c3aed", s=65, zorder=7,
                label="Theorem 2.8 Cost-Nonincreasing Rank-Ascent Path")

    # Plot all 4 extreme points and their costs
    for _idx, (_pt, _cost) in enumerate(zip(_ext_pts6, _ext_costs6)):
        _is_opt = _opt_ext_mask[_idx]
        ax6.scatter(*_pt, s=130 if _is_opt else 80, color="#16a34a" if _is_opt else "#1e293b",
                    edgecolor="black", zorder=8)
        ax6.text(_pt[0] + 0.08, _pt[1] + 0.08, f"$w^{_idx+1}=({_pt[0]:.0f},{_pt[1]:.0f})$\n$c'w^{_idx+1}={_cost:.2f}$",
                 fontsize=8.5, weight="bold" if _is_opt else "normal",
                 color="#15803d" if _is_opt else "#334155")

    ax6.scatter(*opt_x6, s=180, marker="*", color="#f59e0b", edgecolor="black", zorder=9,
                label=f"Optimal Extreme Point: ({opt_x6[0]:.1f}, {opt_x6[1]:.1f})")

    ax6.set_xlim(-0.6, 4.2)
    ax6.set_ylim(-0.6, 4.6)
    ax6.set_xlabel("$x_1$")
    ax6.set_ylabel("$x_2$")
    ax6.set_title(rf"Theorem 2.7 & 2.8 ($\theta={theta_deg_slider.value}^\circ$): $c=({c_sec6[0]:.2f}, {c_sec6[1]:.2f})'$, Optimal Value $v^* = {v_opt6:.3f}$")
    ax6.legend(loc="upper right", fontsize=7.8)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()

    _df_ext6 = pl.DataFrame({
        "Extreme Point w^i": [f"w^{i+1} = ({p[0]:.0f}, {p[1]:.0f})" for i, p in enumerate(_ext_pts6)],
        "Objective Cost c'w^i": np.round(_ext_costs6, 4),
        "In Optimal Face Q (c'w^i == v*)?": ["✅ YES (Optimal Extreme Point)" if m else "No" for m in _opt_ext_mask],
    })

    mo.hstack([
        fig6,
        mo.vstack([
            mo.md(
                f"### Finite Extreme-Point Enumeration vs. Optimal Solution\n"
                f"- **Cost Vector:** $c = ({c_sec6[0]:.3f}, {c_sec6[1]:.3f})'$\n"
                f"- **Optimal Corner:** $x^* = ({opt_x6[0]:.1f}, {opt_x6[1]:.1f})'$ with $v^* = {v_opt6:.4f}$\n"
                f"- **Number of Optimal Extreme Points in $Q$:** **{int(np.sum(_opt_ext_mask))}** "
                f"{'(Multiple optima along an edge—yet extreme points of $Q$ are extreme points of $P$!)' if np.sum(_opt_ext_mask) > 1 else '(Unique vertex optimum!)'}"
            ),
            _df_ext6,
        ]),
    ], widths=[0.56, 0.44])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Representation of Bounded Polyhedra* (Convex Hull of Extreme Points)
    """)
    return


@app.cell(hide_code=True)
def sec7_theory(mo):
    mo.md(r"""
    ### 7.1 Formal Theory: External ($\mathcal{H}$) vs. Internal ($\mathcal{V}$) Representation

    Up to now, we have described every polyhedron **from the outside** by listing its bounding hyperplanes/halfspaces:

    $$
    P = \{x \in \mathbb{R}^n \mid Ax \ge b\} \qquad \text{(Inequality / }\mathcal{H}\text{-Representation)}
    $$

    Theorem 2.9 establishes a dual geometric perspective: any nonempty **bounded** polyhedron (polytope) can be completely described **from the inside** as the set of all convex combinations of its extreme points!

    #### **Theorem 2.9 (Minkowski-Weyl / Krein-Milman Representation of Bounded Polyhedra)**
    **A nonempty and bounded polyhedron $P \subset \mathbb{R}^n$ is the convex hull of its extreme points:**

    $$
    \boxed{P = \operatorname{conv}(\{x^1, x^2, \dots, x^K\}) = \left\{ \sum_{i=1}^K \mu_i x^i \;\middle|\; \mu_i \ge 0 \text{ for all } i,\; \sum_{i=1}^K \mu_i = 1 \right\}}
    $$

    where $\{x^1, \dots, x^K\}$ is the finite set of extreme points of $P$.

    #### **Constructive Proof by Induction on Dimension *(Ray-Shooting from an Extreme Point)***

    **Part 1 ($\operatorname{conv}(\{x^1, \dots, x^K\}) \subseteq P$):** Since each extreme point $x^i \in P$ and $P$ is a convex set (Theorem 2.1(b)), every convex combination of $x^1, \dots, x^K$ belongs to $P$ (Theorem 2.1(c)).

    **Part 2 ($P \subseteq \operatorname{conv}(\{x^1, \dots, x^K\})$):** Define $\dim(P)$ as the smallest integer $k$ such that $P$ is contained in a $k$-dimensional affine subspace $S \subseteq \mathbb{R}^n$. We induct on $k = \dim(P)$:

    * **Base Case ($k = 0$):** A $0$-dimensional polyhedron consists of a single point $P = \{x^1\}$, which is trivially an extreme point of $P$ ($z = 1 \cdot x^1$).
    * **Induction Hypothesis:** Assume every nonempty bounded polyhedron of dimension $\le k - 1$ is the convex hull of its extreme points.
    * **Inductive Step ($\dim(P) = k$):** Take any point $z \in P$. If $z$ is already an extreme point of $P$, we are done. Otherwise, pick **any extreme point $y$ of $P$** (guaranteed to exist by Corollary 2.2 because $P$ is nonempty and bounded!) and shoot a half-line starting at $z$ in the direction away from $y$:

    $$
    u(\lambda) = z + \lambda(z - y), \qquad \lambda \ge 0
    $$

    **Ray Hits a Boundary Face $Q$:** Because $P$ is **bounded**, this half-line cannot stay inside $P$ for arbitrarily large $\lambda$. There is a largest scalar $\lambda^* > 0$ right where the ray is about to exit $P$ and violate some constraint $a_{i^*}'x \ge b_{i^*}$. At $\lambda = \lambda^*$, the boundary hit-point

    $$
    u = z + \lambda^*(z - y) \in P \quad \text{satisfies} \quad a_{i^*}'u = b_{i^*} \quad \text{and} \quad a_{i^*}'(z - y) < 0
    $$

    **Dimension Reduction ($\dim(Q) \le k - 1$):** Define the boundary face $Q = \{x \in P \mid a_{i^*}'x = b_{i^*}\}$. Because $z, y \in S$ (the $k$-dimensional affine subspace containing $P$) and $a_{i^*}'(z - y) < 0$, the new equality constraint $a_{i^*}'x = b_{i^*}$ is linearly independent of the $n - k$ equations defining $S$. Thus $Q$ lies in an affine subspace of dimension **$k - 1$**, so $\dim(Q) \le k - 1$!

    **Apply Induction Hypothesis to $Q$:** Since $u \in Q$ and $\dim(Q) \le k - 1$, $u$ can be written as a convex combination of the extreme points $y^1, \dots, y^r$ of $Q$:

    $$
    u = \sum_{i=1}^r \lambda_i y^i, \qquad \lambda_i \ge 0,\quad \sum_{i=1}^r \lambda_i = 1
    $$

    *(Moreover, every extreme point $y^i$ of $Q$ has $n$ linearly independent active constraints from $P$, so every $y^i$ is also an extreme point of $P$!)*

    **Reconstruct $z$:** From $u = z + \lambda^*(z - y) = (1 + \lambda^*)z - \lambda^* y$, solve directly for $z$:

    $$
    \boxed{z = \frac{u + \lambda^* y}{1 + \lambda^*} = \left(\frac{\lambda^*}{1 + \lambda^*}\right) y + \sum_{i=1}^r \left(\frac{\lambda_i}{1 + \lambda^*}\right) y^i}
    $$

    Notice that all coefficients $\frac{\lambda^*}{1 + \lambda^*}$ and $\frac{\lambda_i}{1 + \lambda^*}$ are nonnegative, and their sum is:

    $$
    \frac{\lambda^*}{1 + \lambda^*} + \sum_{i=1}^r \frac{\lambda_i}{1 + \lambda^*} = \frac{\lambda^* + 1}{1 + \lambda^*} = 1
    $$

    Thus $z$ is a convex combination of extreme points of $P$! $\blacksquare$

    > **Why This Theorem Is So Powerful in Optimization:**
    >
    > 1. **Instant Intuition for LP Optimality:** On a bounded polyhedron $P = \operatorname{conv}(\{x^1, \dots, x^K\})$, any feasible point is $x = \sum_{i=1}^K \mu_i x^i$ ($\mu_i \ge 0, \sum \mu_i = 1$), so its objective value is $c'x = \sum_{i=1}^K \mu_i (c'x^i)$—a weighted average of the corner costs $\{c'x^1, \dots, c'x^K\}$. The minimum of a weighted average is always achieved by placing weight $\mu_{i^*} = 1$ on the single minimum-cost corner $x^{i^*}$!
    > 2. **Foundation of Column Generation & Dantzig-Wolfe Decomposition:** In large-scale optimization, Dantzig-Wolfe decomposition replaces $Ax \ge b$ with convex combinations of extreme points $x = \sum_{i=1}^K \mu_i x^i$!

    ---

    ### 7.2 Plain-English Deconstruction & Notation Lexicon

    | Symbol / Equation | Plain-English Meaning | Role in the Proof of Theorem 2.9 |
    | :--- | :--- | :--- |
    | $u(\lambda) = z + \lambda(z - y),\; \lambda \ge 0$ | **Ray from $y$ through $z$:** Start at interior point $z$ and walk in the direction $z - y$ (directly away from corner $y$). | Because $P$ is bounded, walking in a straight line must eventually hit a boundary wall of $P$. |
    | $\lambda^* = \min_{i:\, a_i'(z-y) < 0} \frac{a_i'z - b_i}{-a_i'(z-y)}$ | **Minimum Ratio Test for Ray Exit:** The exact step size $\lambda^* > 0$ where the ray first hits a bounding hyperplane $a_{i^*}'x = b_{i^*}$. | Gives the boundary point $u = z + \lambda^*(z - y)$ lying on the lower-dimensional facet $Q$. |
    | $z = \frac{1}{1+\lambda^*} u + \frac{\lambda^*}{1+\lambda^*} y$ | **Reverse Interpolation:** Since $u$ is obtained by extending the segment $[y, z]$ past $z$, $z$ lies strictly *between* $y$ and $u$! | Expresses $z$ as a convex combination of the 0D corner $y$ and the $(k-1)$D boundary point $u \in Q$. |

    ---

    ### 7.3 Worked Analytical Examples

    #### **Example A: The 3D Simplex in $\mathbb{R}^3$**
    Consider the 3D simplex $P = \{(x_1, x_2, x_3) \in \mathbb{R}^3 \mid x_1 + x_2 + x_3 \le 1,\; x_1, x_2, x_3 \ge 0\}$. It has $K = 4$ extreme points:

    $$
    x^1 = (1, 0, 0)', \quad x^2 = (0, 1, 0)', \quad x^3 = (0, 0, 1)', \quad x^4 = (0, 0, 0)'
    $$

    Take any point $z = (1/3,\, 1/3,\, 1/4)' \in P$ (since $1/3 + 1/3 + 1/4 = 11/12 \le 1$). Its unique representation as a convex combination of $x^1, x^2, x^3, x^4$ is:

    $$
    z = \frac{1}{3}\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + \frac{1}{3}\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} + \frac{1}{4}\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} + \frac{1}{12}\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \qquad \text{where } \frac{1}{3} + \frac{1}{3} + \frac{1}{4} + \frac{1}{12} = 1.
    $$

    #### **Example B: Step-by-Step Ray-Shooting Construction in $\mathbb{R}^2$**
    In our 2D polygon $P$ with extreme points $w^1=(0,0)'$, $w^2=(2,0)'$, $w^3=(3,1)'$, $w^4=(0,4)'$, take the interior point $z = (1.5, 1.0)'$ and anchor extreme point $y = w^1 = (0, 0)'$:

    1. **Direction vector:** $z - y = (1.5, 1.0)'$.
    2. **Ray:** $u(\lambda) = (1.5, 1.0)' + \lambda(1.5, 1.0)' = (1 + \lambda)(1.5, 1.0)'$.
    3. **Hit boundary wall:** Check when $u(\lambda)$ hits $x_1 + x_2 = 4$ (Constraint 3) vs. $x_1 - x_2 = 2$ (Constraint 4): Constraint 3 gives $1.5(1+\lambda) + 1.0(1+\lambda) = 2.5(1+\lambda) = 4 \implies 1 + \lambda = 1.6 \implies \mathbf{\lambda^* = 0.6}$, while Constraint 4 gives $0.5(1+\lambda) = 2 \implies 1 + \lambda = 4.0 \implies \lambda = 3.0$. The ray hits Constraint 3 first at $\lambda^* = 0.6$, giving boundary point $u = 1.6 \times (1.5, 1.0)' = \mathbf{(2.4, 1.6)'}$ on the edge $[w^3, w^4]$!
    4. **Express $u = (2.4, 1.6)'$ on the 1D edge joining $w^3 = (3, 1)'$ and $w^4 = (0, 4)'$:** Since $2.4 = 0.8(3) + 0.2(0)$ and $1.6 = 0.8(1) + 0.2(4)$, we have $u = 0.8 w^3 + 0.2 w^4$.
    5. **Reconstruct $z$ via Theorem 2.9 formula ($1 + \lambda^* = 1.6$):**

    $$
    z = \frac{0.6}{1.6} w^1 + \frac{0.8}{1.6} w^3 + \frac{0.2}{1.6} w^4 = \mathbf{0.375\, w^1 + 0.500\, w^3 + 0.125\, w^4} = 0.375\begin{bmatrix}0\\0\end{bmatrix} + 0.5\begin{bmatrix}3\\1\end{bmatrix} + 0.125\begin{bmatrix}0\\4\end{bmatrix} = \begin{bmatrix}1.5\\1.0\end{bmatrix}!
    $$
    """)
    return


@app.cell
def sec7_ui(mo):
    z1_slider = mo.ui.slider(start=0.3, stop=2.2, step=0.1, value=1.5, label="Interior point $z_1$:")
    z2_slider = mo.ui.slider(start=0.3, stop=1.8, step=0.1, value=1.0, label="Interior point $z_2$:")
    anchor_y_dropdown = mo.ui.dropdown(
        options={
            "w^1 = (0, 0)": 0,
            "w^2 = (2, 0)": 1,
            "w^3 = (3, 1)": 2,
            "w^4 = (0, 4)": 3,
        },
        value="w^1 = (0, 0)",
        label="Anchor Extreme Point $y$ for Ray-Shooting:"
    )
    mo.hstack([z1_slider, z2_slider, anchor_y_dropdown])
    return anchor_y_dropdown, z1_slider, z2_slider


@app.cell
def sec7_code(
    A_sec2,
    anchor_y_dropdown,
    b_sec2,
    linprog,
    mo,
    np,
    pl,
    plt,
    z1_slider,
    z2_slider,
):
    # Interactive Theorem 2.9 Ray-Shooting Decomposition + Master Convex-Hull LP
    W_ext = np.array([
        [0.0, 0.0],  # w^1
        [2.0, 0.0],  # w^2
        [3.0, 1.0],  # w^3
        [0.0, 4.0],  # w^4
    ])
    z_pt = np.array([z1_slider.value, z2_slider.value])
    y_idx = anchor_y_dropdown.value
    y_pt = W_ext[y_idx]

    # Step 1: Shoot ray u(lam) = z + lam * (z - y), lam >= 0 until hitting a wall of A_sec2 x >= b_sec2
    _dir_zy = z_pt - y_pt
    _slacks_z = A_sec2 @ z_pt - b_sec2
    _rates_zy = A_sec2 @ _dir_zy

    _cand_lams = []
    for _i in range(4):
        if _rates_zy[_i] < -1e-9:
            _cand_lams.append((-_slacks_z[_i] / _rates_zy[_i], _i))
    _lam_star, _hit_wall = min(_cand_lams, key=lambda t: t[0])
    u_hit = z_pt + _lam_star * _dir_zy

    # Step 2: Use SciPy (HiGHS) to decompose u_hit as a convex combination of the extreme points on that facet
    _A_eq7 = np.vstack([W_ext.T, np.ones((1, 4))])
    _b_eq7 = np.append(u_hit, 1.0)
    _bounds7 = []
    for _k in range(4):
        if _k == y_idx or abs(A_sec2[_hit_wall] @ W_ext[_k] - b_sec2[_hit_wall]) > 1e-6:
            _bounds7.append((0.0, 0.0))
        else:
            _bounds7.append((0.0, None))
    _res7 = linprog(np.zeros(4), A_eq=_A_eq7, b_eq=_b_eq7, bounds=_bounds7, method="highs")
    lam_u_vals = _res7.x.copy()

    # Step 3: Theorem 2.9 Inductive Formula: z = (lam* / (1 + lam*)) y + sum_i (lam_i / (1 + lam*)) y^i
    mu_thm29 = lam_u_vals / (1.0 + _lam_star)
    mu_thm29[y_idx] += _lam_star / (1.0 + _lam_star)

    fig7, ax7 = plt.subplots(figsize=(7.4, 4.7), dpi=110)
    ax7.fill(W_ext[:, 0], W_ext[:, 1], color="#f8fafc", edgecolor="#334155", lw=2.0, label=r"$P = \mathrm{conv}(\{w^1, w^2, w^3, w^4\})$")

    # Plot extreme points
    for _k, _w in enumerate(W_ext):
        ax7.scatter(*_w, s=100, color="#16a34a" if mu_thm29[_k] > 1e-6 else "#94a3b8", edgecolor="black", zorder=6)
        ax7.text(_w[0] + 0.08, _w[1] + 0.08, f"$w^{_k+1}=({_w[0]:.0f},{_w[1]:.0f})$\n$\\mu_{_k+1}={mu_thm29[_k]:.3f}$",
                 fontsize=8.5, weight="bold" if mu_thm29[_k] > 1e-6 else "normal",
                 color="#15803d" if mu_thm29[_k] > 1e-6 else "#64748b")

    # Plot ray from y through z to u_hit on facet Q
    ax7.plot([y_pt[0], z_pt[0]], [y_pt[1], z_pt[1]], color="#2563eb", lw=2.4, label="Segment $[y, z]$")
    ax7.annotate("", xy=u_hit, xytext=z_pt,
                 arrowprops=dict(arrowstyle="->", lw=2.5, ls="--", color="#dc2626"))
    ax7.scatter(*y_pt, s=140, color="#2563eb", edgecolor="black", zorder=7, label=f"Anchor Extreme Point $y = w^{y_idx+1}$")
    ax7.scatter(*z_pt, s=130, color="#f59e0b", edgecolor="black", zorder=8, label=f"Target Point $z = ({z_pt[0]:.1f}, {z_pt[1]:.1f})$")
    ax7.scatter(*u_hit, s=120, marker="D", color="#dc2626", edgecolor="black", zorder=8,
                label=rf"Boundary Hit $u = z + \lambda^*(z-y) = ({u_hit[0]:.2f}, {u_hit[1]:.2f})$ ($\lambda^*={_lam_star:.2f}$)")

    ax7.set_xlim(-0.5, 4.1)
    ax7.set_ylim(-0.5, 4.5)
    ax7.set_xlabel("$x_1$")
    ax7.set_ylabel("$x_2$")
    ax7.set_title("Theorem 2.9: Constructive Ray-Shooting Convex Hull Representation")
    ax7.legend(loc="upper right", fontsize=8)
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()

    _df_thm29 = pl.DataFrame({
        "Extreme Point": [f"w^{i+1} = ({W_ext[i,0]:.0f}, {W_ext[i,1]:.0f})" for i in range(4)],
        "Facet Weight λ_i for u": np.round(lam_u_vals, 4),
        "Final Weight μ_i for z": np.round(mu_thm29, 4),
        "Contribution μ_i * w^i": [f"({(mu_thm29[i]*W_ext[i,0]):.3f}, {(mu_thm29[i]*W_ext[i,1]):.3f})" for i in range(4)],
    })

    mo.hstack([
        fig7,
        mo.vstack([
            mo.md(
                f"### Inductive Ray-Shooting Breakdown (Theorem 2.9)\n"
                f"1. **Ray Exit Scalar:** $\\lambda^* = {_lam_star:.4f}$ hits facet $Q$ (Constraint {_hit_wall+1}) at $u = ({u_hit[0]:.3f}, {u_hit[1]:.3f})'$\n"
                f"2. **Reverse Formula:** $z = \\frac{{\\lambda^*}}{{1+\\lambda^*}} y + \\sum_{{i=1}}^4 \\frac{{\\lambda_i}}{{1+\\lambda^*}} w^i$\n"
                f"3. **Reconstruction Check:** $\\sum_{{i=1}}^4 \\mu_i w^i = ({(mu_thm29 @ W_ext)[0]:.3f}, {(mu_thm29 @ W_ext)[1]:.3f})' = z$, $\\sum \\mu_i = {mu_thm29.sum():.4f}$"
            ),
            _df_thm29,
        ]),
    ], widths=[0.55, 0.45])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 02: The Simplex Method and Duality Theory
    """)
    return


@app.cell(hide_code=True)
def header_mod2(mo):
    mo.md(r"""
    ### *Algorithmic Edge-Walking, Tableau Mechanics, and the Economic & Computational Power of Duality*

    ---

    ### The Bridge from Geometry to Computation & Economics

    In **Module 01**, we proved that an optimal solution to any linear program always exists at an extreme point (BFS) of the feasible polyhedron. But how do we find that optimal corner efficiently when there are millions of possibilities? And once found, what is the economic and computational value of the resulting solution?

    * **Part 1 (The Simplex Method):** Turns polyhedral geometry into an active algorithm. By walking along 1-dimensional edges from corner to adjacent corner in cost-reducing directions, Simplex systematically traverses the boundary of the feasible polyhedron, reaching optimality in typically $2m$ to $3m$ pivots.
    * **Part 2 (Duality Theory):** Uncovers the twin optimization problem hiding behind every linear program. Duality provides instant certificates of global optimality (zero duality gap), marginal shadow prices ($\frac{\partial v^*}{\partial b_i} = p_i^*$), computational speedups when $m \gg n$, and the mathematical backbone of modern Branch-and-Bound and Stochastic Programming.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Module 02 — Topic 1: The Simplex Method (Mechanics, Geometry & Pivoting Algorithms)
    """)
    return


@app.cell(hide_code=True)
def sec9_theory(mo):
    mo.md(r"""
    ### 1.1 The Geometric & Intuitive Foundation: Edge-Walking on Polyhedra

    In **Module 01**, we proved the **Fundamental Theorem of Linear Programming**: if a standard form LP has an optimal solution, at least one **Basic Feasible Solution (BFS)** (extreme point / corner) is optimal.
    However, a polyhedron can have up to $\binom{n}{m}$ basic solutions. For a modest LP with $n = 100$ variables and $m = 20$ constraints, $\binom{100}{20} \approx 5.35 \times 10^{20}$ candidates—far too many to enumerate!

    The **Simplex Method**, invented by George Dantzig in 1947, does not search blindly. Instead, it starts at an initial BFS and **walks along 1-dimensional edges of the polyhedron from one BFS to an adjacent BFS, strictly improving (or maintaining) the objective value at every step**, until no further improvement is possible.

    ---

    ### 1.2 Algebraic Derivation of Simplex: Directions, Reduced Costs, and Optimality

    Consider a standard form linear program:

    $$
    \begin{aligned}
    \min_{x \in \mathbb{R}^n} \quad & c'x \\
    \text{subject to} \quad & Ax = b \\
    & x \ge 0
    \end{aligned}
    $$

    where $A \in \mathbb{R}^{m \times n}$ has full row rank $m$.

    #### **1. Moving Along Feasible Basic Directions**
    Suppose we currently stand at a basic feasible solution $x \in \mathbb{R}^n$ associated with basis matrix $B = [A_{B(1)} \cdots A_{B(m)}] \in \mathbb{R}^{m \times m}$. Partition variables into basic variables $x_B = B^{-1}b \ge 0$ and nonbasic variables $x_N = 0$.

    To move away from $x$ along a direction $d \in \mathbb{R}^n$ ($x_{\text{new}} = x + \theta d$ with $\theta > 0$), we increase **one nonbasic variable $x_j$** ($j \notin B$) at unit rate ($d_j = 1$) and keep all other nonbasic variables locked at zero ($d_k = 0$ for $k \notin B,\, k \ne j$).

    To stay feasible ($A(x + \theta d) = b$), since $Ax = b$ and $\theta > 0$, we need $Ad = 0$. Splitting $Ad = \sum_{k=1}^n A_k d_k$ into the basic columns $B$ and the single active nonbasic column $A_j$ (all other $d_k = 0$):

    $$
    Ad = B d_B + A_j d_j = 0 \quad \xrightarrow{d_j = 1} \quad B d_B = -A_j \iff \boxed{d_B = -B^{-1}A_j}
    $$

    Putting all components together gives the **$j$-th Basic Direction** $d \in \mathbb{R}^n$:

    $$
    d_k = \begin{cases} (-B^{-1}A_j)_i & \text{if } k = B(i) \text{ is the } i\text{-th basic variable} \\ 1 & \text{if } k = j \text{ (the entering variable)} \\ 0 & \text{if } k \notin B \text{ and } k \ne j \end{cases}
    $$

    In block-matrix form (partitioned into basic $B$, entering $j$, and remaining nonbasic $N \setminus \{j\}$):

    $$
    x_{\text{new}} = x + \theta d = \begin{bmatrix} x_B \\ 0 \\ 0 \end{bmatrix} + \theta \begin{bmatrix} -B^{-1}A_j \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} x_B - \theta B^{-1}A_j \\ \theta \\ 0 \end{bmatrix} \implies A x_{\text{new}} = \begin{bmatrix} B & A_j & A_{N \setminus \{j\}} \end{bmatrix} \begin{bmatrix} x_B - \theta B^{-1}A_j \\ \theta \\ 0 \end{bmatrix} = B x_B - \theta A_j + \theta A_j = b
    $$

    #### **2. The Reduced Cost: How Much Does the Objective Change?**
    If we step along the basic direction $d$ by an amount $\theta > 0$, the objective function changes at the rate $c'd$:

    $$
    \text{Rate of Cost Change} = c'd = c_B' d_B + c_j d_j = c_j - c_B' B^{-1} A_j
    $$

    We define this rate as the **Reduced Cost** of variable $x_j$:

    $$
    \boxed{\bar{c}_j \equiv c_j - c_B' B^{-1} A_j = c_j - p' A_j}
    $$

    where $p' \equiv c_B' B^{-1} \in \mathbb{R}^{1 \times m}$ is the vector of **Simplex Multipliers** (which will become the **Dual Variables** in Topic 2!).

    **Key Property:** For every **basic variable** $x_{B(i)}$, its reduced cost is **identically zero**:

    $$
    \bar{c}_{B(i)} = c_{B(i)} - c_B' B^{-1} A_{B(i)} = c_{B(i)} - c_B' e_i = c_{B(i)} - c_{B(i)} = 0
    $$

    **Physical Meaning:** $\bar{c}_j$ is the net marginal cost of producing one unit of activity $j$:

    * $c_j$ = direct objective cost of activity $j$.
    * $p'A_j = \sum_{i=1}^m p_i a_{ij}$ = imputed economic value of the $m$ resources consumed by activity $j$.
    * If $\bar{c}_j < 0$, activity $j$ delivers more value than the resources it consumes $\implies$ **increasing $x_j$ decreases total cost!**

    #### **3. Theorem 3.1 (Simplex Optimality Conditions)**
    Consider a basic feasible solution $x$ with basis matrix $B$ and reduced cost vector $\bar{c} = c' - c_B' B^{-1} A$:

    **Condition 1: If $\bar{c} \ge 0$ (all reduced costs are nonnegative), then $x$ is an OPTIMAL SOLUTION.**

    *Proof:* For any feasible vector $y \ge 0$ with $Ay = b$:

    $$
    c'y - c'x = c'(y - x) = (c' - c_B' B^{-1}A)(y - x) + c_B' B^{-1}(Ay - Ax) = \bar{c}'(y - x) + c_B' B^{-1}(b - b) = \bar{c}'(y - x)
    $$

    Since $x_N = 0$ at the current BFS, $y_j - x_j = y_j \ge 0$ for all nonbasic $j$, and $\bar{c}_B = 0$ for all basic variables. Therefore:

    $$
    c'y - c'x = \sum_{j \notin B} \bar{c}_j y_j \ge 0 \implies c'y \ge c'x \quad \forall y \in P. \quad \blacksquare
    $$

    **Condition 2: If $\bar{c}_j > 0$ strictly for all nonbasic variables, then $x$ is the UNIQUE optimal solution.**

    #### **4. The Minimum Ratio Test: Step Size $\theta^*$ & Leaving Variable**
    If there exists some nonbasic variable $x_j$ with $\bar{c}_j < 0$, we choose $x_j$ to **enter the basis**.
    As $\theta$ increases, the new point is $x(\theta) = x + \theta d$:

    * Nonbasic variables: $x_j(\theta) = \theta \ge 0$, other nonbasics $= 0$.
    * Basic variables: $x_B(\theta) = x_B - \theta u$, where $u \equiv B^{-1}A_j \in \mathbb{R}^m$.

    How large can $\theta$ become before some basic variable violates nonnegativity ($x_{B(i)} < 0$)?

    * If $u_i \le 0$, then $x_{B(i)}(\theta) = x_{B(i)} - \theta u_i \ge x_{B(i)} \ge 0$ for all $\theta \ge 0$ (the component never decreases).
    * **Unboundedness Test:** If **$u = B^{-1}A_j \le 0$** (every component non-positive), then $\theta$ can grow to $+\infty$ with cost decreasing at rate $\bar{c}_j < 0$. **The optimal cost is $-\infty$ (Unbounded LP)!**
    * If some $u_i > 0$, then $x_{B(i)}(\theta) \ge 0 \iff \theta \le \frac{x_{B(i)}}{u_i}$. Therefore, the maximum feasible step size is:

    $$
    \boxed{\theta^* = \min_{\substack{i = 1, \dots, m \\ u_i > 0}} \frac{x_{B(i)}}{u_i}}
    $$

    Let $l \in \{1, \dots, m\}$ be an index that achieves this minimum ($u_l > 0$ and $\theta^* = x_{B(l)} / u_l$). The variable **$x_{B(l)}$ drops to zero and LEAVES the basis**, replaced by $x_j$!

    ---

    ### 1.3 Degeneracy, Stalling, and Anti-Cycling Rules

    What happens if $\theta^* = 0$?
    This occurs when the current BFS is **degenerate**, meaning some basic variable $x_{B(l)} = 0$ with $u_l > 0$.
    Then $\theta^* = 0 / u_l = 0$:

    * The basis changes from $B$ to a new basis $\bar{B}$ (swapping $x_{B(l)} \leftrightarrow x_j$).
    * But the physical solution vector **does not move at all**: $x_{\text{new}} = x + 0 \cdot d = x$.
    * The objective value **does not change**: $c'x_{\text{new}} = c'x$.

    #### **The Danger of Cycling (Alan Hoffman's Counterexample, 1953)**
    If a sequence of pivots all have $\theta^* = 0$, the simplex method can theoretically visit a sequence of bases $B_1 \to B_2 \to \cdots \to B_k \to B_1$, **cycling indefinitely without terminating!**

    #### **Guaranteed Anti-Cycling Pivoting Rules:**

    **1. Bland's Smallest-Index Rule (1977):**

    * **Pivot In:** Among all candidates with $\bar{c}_j < 0$, pick the one with the **smallest variable index $j$**: $j = \min \{k \mid \bar{c}_k < 0\}$.
    * **Pivot Out:** Among all basic variables that tie for the minimum ratio $\theta^* = x_{B(i)} / u_i$, pick the one with the **smallest variable index $B(i)$**: $l = \min \{B(i) \mid u_i > 0,\, x_{B(i)}/u_i = \theta^*\}$.
    * Under Bland's rule, the Simplex method **never cycles** and is guaranteed to terminate in a finite number of steps!

    **2. Lexicographic / Perturbation Method:**

    * Perturb the right-hand side $b(\epsilon) = b + \sum_{k=1}^m \epsilon^k A_k$. This breaks all ties lexicographically, guaranteeing $\theta^* > 0$ in the perturbed problem so cycling is impossible.

    ---

    ### 1.4 Finding an Initial BFS: The Two-Phase Simplex Method

    How do we start the Simplex method if an initial BFS is not immediately obvious (e.g. constraints have $\ge$ or $=$ signs)?

    **1. Phase I (Drive Infeasibility to Zero):**

    * Multiply rows by $-1$ if necessary so $b \ge 0$.
    * Add $m$ non-negative **artificial variables** $y = (y_1, \dots, y_m)' \ge 0$:

    $$
    \min_{x \ge 0,\, y \ge 0} \quad \sum_{i=1}^m y_i \quad \text{subject to} \quad Ax + y = b
    $$

    * An initial BFS is immediately obvious: set $x = 0$ (nonbasic) and $y = b \ge 0$ (basic, with $B = I_m$). Solve with Simplex. At optimality: if optimal cost $\sum y_i^* > 0$, **the original linear program is INFEASIBLE** (no solution satisfies $Ax = b, x \ge 0$); if optimal cost $\sum y_i^* = 0$, all artificial variables are zero—drop the artificial columns to obtain an initial BFS for the original problem.

    **2. Phase II (Optimize Original Objective):**

    * Starting from the clean BFS produced by Phase I, optimize the true objective $c'x$.
    """)
    return


@app.cell(hide_code=True)
def _(linprog, mo, np, pl):
    # Worked 3x7 Numerical Example: Connecting p' = c_B' B^{-1}, Reduced Costs c_bar_j = c_j - p' A_j, and Simplex Pivots
    _A37 = np.array([
        [1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [2.0, 1.0, 3.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0],
    ])  # shape: (3, 7)
    _b37 = np.array([10.0, 15.0, 8.0])
    _c37 = np.array([-4.0, -3.0, -5.0, -2.0, 0.0, 0.0, 0.0])

    # Trace full Simplex iterations using exact matrix formulas
    _basis = [4, 5, 6]  # 0-indexed columns {5, 6, 7} (slacks I_3)
    _iter_rows = []
    _snap_iter1 = None

    for _it in range(10):
        _B = _A37[:, _basis]
        _B_inv = np.linalg.inv(_B)
        _x_B = _B_inv @ _b37
        _c_B = _c37[_basis]
        _p = _c_B @ _B_inv                      # Simplex multipliers: p' = c_B' B^{-1}
        _cbar = _c37 - _p @ _A37                # Reduced costs: c_bar_j = c_j - p' A_j

        _x_full = np.zeros(7)
        _x_full[_basis] = _x_B
        _obj = float(_c37 @ _x_full)

        _neg = [j for j in range(7) if _cbar[j] < -1e-9]
        if not _neg:
            _iter_rows.append({
                "Iter": _it,
                "Basis (1-based)": str([i + 1 for i in _basis]),
                "Simplex Multipliers p'": str(np.round(_p, 3).tolist()),
                "Entering": "— (Optimal!)",
                "Leaving": "—",
                "Step θ*": 0.0,
                "Objective c'x": round(_obj, 3),
            })
            break

        _enter = min(_neg, key=lambda j: _cbar[j])  # Most negative reduced cost
        _u = _B_inv @ _A37[:, _enter]               # u = B^{-1} A_j  (so d_B = -u)
        _ratios = [(_x_B[i] / _u[i], i) for i in range(3) if _u[i] > 1e-9]
        _theta, _leave_pos = min(_ratios, key=lambda t: t[0])
        _leave = _basis[_leave_pos]

        if _it == 1:
            _snap_iter1 = {
                "basis": [i + 1 for i in _basis],
                "B": _B.copy(),
                "B_inv": _B_inv.copy(),
                "x_B": _x_B.copy(),
                "c_B": _c_B.copy(),
                "p": _p.copy(),
                "cbar": _cbar.copy(),
                "enter": _enter + 1,
                "u": _u.copy(),
                "d_B": -_u.copy(),
                "theta": float(_theta),
                "leave": _leave + 1,
                "obj": _obj,
            }

        _iter_rows.append({
            "Iter": _it,
            "Basis (1-based)": str([i + 1 for i in _basis]),
            "Simplex Multipliers p'": str(np.round(_p, 3).tolist()),
            "Entering": f"x_{_enter+1} (c̄={_cbar[_enter]:.3f})",
            "Leaving": f"x_{_leave+1}",
            "Step θ*": round(float(_theta), 3),
            "Objective c'x": round(_obj, 3),
        })
        _basis[_leave_pos] = _enter

    _res37 = linprog(_c37, A_eq=_A37, b_eq=_b37, bounds=(0.0, None), method="highs")

    _md_37 = mo.md(
        r"""
    ### 1.2b Worked $3 \times 7$ Numerical Example: Connecting $p' = c_B'B^{-1}$, $\bar{c}_j = c_j - p'A_j$, and Simplex Pivots

    Consider $\min c'x$ subject to $Ax = b,\, x \ge 0$ with $m = 3$ constraints and $n = 7$ variables (slacks $x_5, x_6, x_7$):

    $$
    A = \begin{bmatrix} 1 & 2 & 1 & 1 & 1 & 0 & 0 \\ 2 & 1 & 3 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 2 & 0 & 0 & 1 \end{bmatrix}, \quad
    b = \begin{bmatrix} 10 \\ 15 \\ 8 \end{bmatrix}, \quad
    c' = \begin{bmatrix} -4 & -3 & -5 & -2 & 0 & 0 & 0 \end{bmatrix}
    $$

    * **Iteration 0 (Initial Basis $B = \{5, 6, 7\} \implies B = I_3$):** $x_B = (10, 15, 8)'$, $c_B = (0, 0, 0)'$, so $p' = c_B'B^{-1} = (0, 0, 0)$ and $\bar{c}_j = c_j$. Most negative is $\bar{c}_3 = -5 \implies$ **$x_3$ enters**. Ratio test with $u = B^{-1}A_3 = (1, 3, 1)'$ gives $\theta^* = \min(10/1,\, 15/3,\, 8/1) = 5 \implies$ **$x_6$ leaves**.
    * **Iteration 1 (Basis $B = \{5, 3, 7\}$, where $B \ne I_3$):**
      1. **Basis & Inverse:** $B = [A_5\; A_3\; A_7] = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 1 & 1 \end{bmatrix} \implies B^{-1} = \begin{bmatrix} 1 & -1/3 & 0 \\ 0 & 1/3 & 0 \\ 0 & -1/3 & 1 \end{bmatrix}$, $x_B = B^{-1}b = (5, 5, 3)'$, cost $c'x = -25$.
      2. **Simplex Multipliers ($p' = c_B' B^{-1}$):** Since $c_B' = [0, -5, 0]$, we get $p' = [0, -5, 0] B^{-1} = [0,\; -5/3,\; 0]$.
      3. **Reduced Costs ($\bar{c}_j = c_j - p'A_j$):** For $x_2$, $p'A_2 = (-5/3)(1) = -5/3$, so $\bar{c}_2 = -3 - (-5/3) = -4/3 < 0$ (most negative $\implies$ **$x_2$ enters**). Similarly $\bar{c}_1 = -2/3$, $\bar{c}_4 = -1/3$, $\bar{c}_6 = +5/3$, and basic variables $\{5, 3, 7\}$ have $\bar{c} = 0$.
      4. **Basic Direction & Ratio Test:** $u = B^{-1}A_2 = (5/3, 1/3, 2/3)'$, so $d_B = -u = (-5/3, -1/3, -2/3)'$, and $\theta^* = \min(\frac{5}{5/3}, \frac{5}{1/3}, \frac{3}{2/3}) = 3 \implies$ **$x_5$ leaves**.
    """
        + f"\n**Live Solver Verification:** Optimal objective = `{_res37.fun:.3f}` at $x^* = {np.round(_res37.x, 3).tolist()}$ (matches Simplex Iteration {_iter_rows[-1]['Iter']}!)."
    )

    _df_37 = pl.DataFrame(_iter_rows)
    mo.vstack([_md_37, _df_37])

    return


@app.cell
def sec9_ui(mo):
    simplex_step_slider = mo.ui.slider(
        start=0, stop=3, step=1, value=0,
        label="Simplex Iteration Step (0 = Initial BFS, 1 = First Pivot, 2 = Second Pivot, 3 = Optimal BFS):"
    )
    simplex_step_slider
    return (simplex_step_slider,)


@app.cell
def sec9_code(linprog, mo, np, pl, plt, simplex_step_slider):
    # Interactive Simplex Tableau & Step-by-Step Edge Walking Laboratory
    # Problem: min -3*x1 - 2*x2 - 4*x3
    # Subject to:
    #  (1) x1 + x2 + 2*x3 + x4             = 8
    #  (2)      x2 + 6*x3      + x5        = 12
    #  (3) x1                       + x6   = 4
    #  (4)      x2                       + x7 = 6
    #  x1, ..., x7 >= 0 (m=4, n=7)

    A_smp = np.array([
        [1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 6.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]) # Constraint matrix for 4 constraints and 7 variables (3 original + 4 slack)
    b_smp = np.array([8.0, 12.0, 4.0, 6.0]) # RHS capacities for 4 constraints
    c_smp = np.array([-3.0, -2.0, -4.0, 0.0, 0.0, 0.0, 0.0]) # Objective: min -3*x1 - 2*x2 - 4*x3

    def run_simplex_step_by_step(A, b, c, initial_basis, max_steps=5):
        """Executes the exact algebraic Simplex algorithm."""
        m, n = A.shape
        basis = list(initial_basis)
        history = []

        for step in range(max_steps):
            B = A[:, basis]
            B_inv = np.linalg.inv(B)
            x_B = B_inv @ b
            c_B = c[basis]

            # Simplex Multipliers (Dual vector) p' = c_B' B^-1
            p = c_B @ B_inv

            # Reduced costs \bar{c}_j = c_j - p' A_j
            reduced_costs = c - p @ A

            # Current full solution x and objective value
            x_curr = np.zeros(n)
            x_curr[basis] = x_B
            obj_val = float(c @ x_curr)

            # Check optimality: all reduced costs >= -1e-9
            is_opt = bool(np.all(reduced_costs >= -1e-9))

            info = {
                "step": step,
                "basis": list(basis),
                "x_B": x_B.copy(),
                "x_full": x_curr.copy(),
                "obj_val": obj_val,
                "p": p.copy(),
                "reduced_costs": reduced_costs.copy(),
                "is_optimal": is_opt,
                "B_inv": B_inv.copy(),
                "B_inv_A": B_inv @ A,
            }

            if is_opt:
                info["entering_var"] = None
                info["leaving_var"] = None
                info["theta_star"] = 0.0
                history.append(info)
                break

            # Entering variable: Bland's rule or most negative reduced cost
            # Pricing rule: choose nonbasic j with most negative \bar{c}_j
            candidates = [j for j in range(n) if reduced_costs[j] < -1e-8 and j not in basis]
            j_enter = min(candidates, key=lambda j: reduced_costs[j])

            # Direction in basic variables: u = B^-1 A_j
            u = B_inv @ A[:, j_enter]

            # Minimum ratio test
            ratios = []
            for i in range(m):
                if u[i] > 1e-9:
                    ratios.append((x_B[i] / u[i], i, basis[i]))
                else:
                    ratios.append((np.inf, i, basis[i]))

            valid_ratios = [(r, i, var) for r, i, var in ratios if r < np.inf]
            if not valid_ratios:
                info["unbounded"] = True
                history.append(info)
                break

            # Minimum ratio \theta^*
            theta_star, min_i, var_leave = min(valid_ratios, key=lambda t: t[0])

            info["entering_var"] = j_enter
            info["leaving_var"] = var_leave
            info["theta_star"] = theta_star
            info["u"] = u.copy()
            info["ratios"] = [r[0] for r in ratios]
            history.append(info)

            # Update basis
            basis[min_i] = j_enter

        return history

    _smp_history = run_simplex_step_by_step(A_smp, b_smp, c_smp, initial_basis=[3, 4, 5, 6])
    _step_idx = min(simplex_step_slider.value, len(_smp_history) - 1)
    _curr = _smp_history[_step_idx]

    # SciPy (HiGHS) verification
    _res_smp = linprog(c_smp, A_eq=A_smp, b_eq=b_smp, bounds=(0.0, None), method="highs")
    opt_smp_opt = _res_smp.x.copy()
    opt_smp_obj = float(_res_smp.fun)

    # Build Full Simplex Tableau Table for current step
    # Tableau layout:
    # Row 0: -c_B' B^-1 b | c_1 - p'A_1 ... c_n - p'A_n
    # Rows 1..m: B^-1 b   | B^-1 A_1   ... B^-1 A_n
    _tab_data = {
        "Row Description": ["Obj Row 0 (Reduced Costs)"] + [f"Row {i+1} (x_{_curr['basis'][i]+1})" for i in range(4)],
        "RHS (Value)": [f"{-_curr['obj_val']:.2f}"] + [f"{_curr['x_B'][i]:.2f}" for i in range(4)],
    }
    for _j in range(7):
        _col_vals = [f"{_curr['reduced_costs'][_j]:.2f}"] + [f"{_curr['B_inv_A'][i, _j]:.2f}" for i in range(4)]
        _tab_data[f"x_{_j+1}"] = _col_vals

    _df_tableau = pl.DataFrame(_tab_data)

    # Visualizing Simplex Trajectory across iterations
    fig_smp, ax_smp = plt.subplots(figsize=(7.2, 4.4), dpi=110)
    # Project onto (x1, x2) plane (setting x3 = (12 - x5 - x2)/6 etc.)
    _path_pts = np.array([h["x_full"][:2] for h in _smp_history])
    ax_smp.plot(_path_pts[:, 0], _path_pts[:, 1], "o-", color="#2563eb", lw=2.5, markersize=8, label="Simplex Iteration Path")

    for _k, _pt in enumerate(_path_pts):
        ax_smp.scatter(*_pt, s=110, color="#dc2626" if _k == _step_idx else "#2563eb", edgecolor="black", zorder=6)
        ax_smp.text(_pt[0] + 0.08, _pt[1] + 0.12, f"Step {_k}: ({_pt[0]:.1f}, {_pt[1]:.1f})\nCost: {_smp_history[_k]['obj_val']:.1f}",
                    fontsize=8.5, weight="bold" if _k == _step_idx else "normal", color="#dc2626" if _k == _step_idx else "#1e293b")

    ax_smp.set_xlim(-0.3, 4.5)
    ax_smp.set_ylim(-0.3, 6.5)
    ax_smp.set_xlabel("$x_1$")
    ax_smp.set_ylabel("$x_2$")
    ax_smp.set_title(rf"Simplex Path: Step {_step_idx}/{len(_smp_history)-1} | Current Cost = {_curr['obj_val']:.2f}")
    ax_smp.grid(True, alpha=0.3)
    ax_smp.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()

    _step_summary_md = f"""
    ### Iteration {_step_idx} Status: {'🎯 **OPTIMAL BASIS!** (All $\\bar{c}_j \\ge 0$)' if _curr['is_optimal'] else '🔄 **SUBOPTIMAL** (Improving pivot available)'}
    - **Current Basis $B$:** Columns $\\{{ {', '.join(f'A_{i+1}' for i in _curr['basis'])} \\}}$
    - **Current Basic Solution:** $x_B = {np.round(_curr['x_B'], 2).tolist()}$, **Cost:** $c'x = {_curr['obj_val']:.2f}$
    - **Simplex Multipliers (Dual):** $p' = c_B' B^{{-1}} = {np.round(_curr['p'], 2).tolist()}$
    """

    if not _curr['is_optimal']:
        _step_summary_md += f"""
    - **Entering Variable:** $x_{_curr['entering_var']+1}$ (most negative reduced cost $\\bar{{c}}_{_curr['entering_var']+1} = {_curr['reduced_costs'][_curr['entering_var']]:.2f} < 0$)
    - **Minimum Ratio Test:** $\\theta^* = {_curr['theta_star']:.2f}$ $\\implies$ Variable **$x_{_curr['leaving_var']+1}$ drops to zero and LEAVES the basis!**
    """
    else:
        _step_summary_md += f"""
    - **Optimal Solution Verification:** Optimal cost = **{opt_smp_obj:.2f}** at $x^* = {np.round(opt_smp_opt, 2).tolist()}$ ✓
    """

    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md(_step_summary_md), mo.md("**Current Full Simplex Tableau:**"), _df_tableau]),
            fig_smp,
        ], widths=[0.58, 0.42])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Module 02 — Topic 2: Duality Theory (Foundations, Theorems & The Universal Toolkit)
    """)
    return


@app.cell(hide_code=True)
def sec10_theory(mo):
    mo.md(r"""
    ### 2.1 Why We Need Duality and In Which Cases We Can Use It

    Duality is arguably the most elegant, powerful, and practical concept in all of mathematical optimization. Every linear program (called the **Primal**) has a twin linear program (called the **Dual**) built from the exact same data ($A, b, c$), but viewed from an opposite, complementary perspective.

    To truly understand duality, we must first answer: **Why do we need it, and in which real-world and algorithmic cases do we use it?**

    ```text
                         ┌────────────────────────────────────────┐
                         │          PRIMAL PROBLEM                │
                         │  min c'x  s.t.  Ax = b,  x >= 0        │
                         │  (Allocation of physical resources)    │
                         └───────────────────┬────────────────────┘
                                             │
                        STRONG DUALITY:      │  Zero Gap: c'x* = p*'b
                        SHADOW PRICES:       │  p_i* = dv*/db_i
                        COMPLEMENTARITY:     │  p_i*(a_i'x - b_i) = 0
                                             │  x_j*(c_j - p'A_j) = 0
                                             ▼
                         ┌────────────────────────────────────────┐
                         │           DUAL PROBLEM                 │
                         │  max p'b  s.t.  p'A <= c'              │
                         │  (Valuation of resources & pricing)    │
                         └────────────────────────────────────────┘
    ```

    #### **Seven Major Cases Where Duality Is Essential:**

    **1. Instant, Unambiguous Certificate of Optimality (Zero Duality Gap):**

    * In numerical computation, when should an algorithm stop? If you only solve the primal, you have a candidate solution $x$, but how do you prove no better solution exists without checking all corners?
    * **Weak Duality** guarantees that **every** dual feasible solution $p$ provides a hard lower bound on the primal: $p'b \le c'x$.
    * **Strong Duality** proves that at optimality, the bound closes completely: **$c'x^* = p^{*\top} b$**! The moment your solver finds a pair $(x, p)$ where $c'x = p'b$, you have a mathematical proof of global optimality.

    **2. Economic Analysis, Marginal Valuation, and Shadow Prices ($\frac{\partial v^*}{\partial b_i} = p_i^*$):**

    * Suppose a factory manager has $b_1 = 100$ hours of machine capacity. The optimal dual multiplier $p_1^* = \$15/\text{hour}$ is the **Shadow Price**.
    * It answers: *"If a supplier offers 10 extra machine hours for \$12/hour, should we buy them?"* Yes! Each extra hour increases profit by \$15, netting \$3/hour in pure profit.
    * If machine 2 has idle capacity left over (slack $> 0$), Complementary Slackness proves its shadow price is **identically $\$0$**—paying even one cent for more of an unused resource is irrational!

    **3. Massive Computational Speedups when $m \gg n$:**

    * The Simplex method's computational burden scales primarily with the number of constraints $m$, not the number of variables $n$.
    * Suppose you face an LP with **$1,000,000$ constraints** and **$50$ variables** ($m = 10^6, n = 50$). Solving this directly in the primal requires inverting $10^6 \times 10^6$ basis matrices—practically impossible!
    * Its **Dual** problem has only **$50$ constraints** and **$1,000,000$ variables** ($m_{\text{dual}} = 50, n_{\text{dual}} = 10^6$). In the dual, the basis matrix is just $50 \times 50$, which Simplex solves in milliseconds!

    **4. The Dual Simplex Method & Mixed-Integer Programming (Branch-and-Bound):**

    * In Mixed-Integer Linear Programming (MIP solvers), the solver creates a search tree by branching on fractional variables (e.g., adding a constraint $x_1 \le 3$).
    * When a new constraint is added to an already-solved LP, the old optimal solution $x^*$ violates the new constraint (**primal infeasible**). However, the old dual multipliers **remain dual feasible**!
    * Instead of restarting from scratch, solvers run the **Dual Simplex Method**, which restores feasibility in just 1 or 2 pivots. Modern MIP solvers spend over 90% of their execution time running Dual Simplex at branch-and-bound nodes!

    **5. Decomposition in Large-Scale and Stochastic Programming:**

    * **Benders Decomposition / L-Shaped Method:** In two-stage stochastic programming under uncertainty, the second-stage recourse subproblems are solved to obtain their dual vectors $p(\xi)$. These dual vectors define **Benders optimality cuts** $E[p(\xi)'h(\xi)] + E[p(\xi)'T(\xi)]x \le \theta$, transferring future risk information back to the first-stage master decision!
    * **Dantzig-Wolfe Decomposition & Column Generation:** In cutting-stock, crew scheduling, and vehicle routing problems, there are exponentially many columns. Dual variables from the restricted master problem are sent to a subproblem ("pricing problem") to generate only the columns with negative reduced cost $\bar{c}_j = c_j - p'A_j < 0$.

    **6. Post-Optimality & Sensitivity Analysis Without Re-solving:**

    * Duality allows us to analytically compute the allowable ranges of objective coefficients $\Delta c$ and right-hand-side capacities $\Delta b$ over which the current optimal basis remains valid.

    **7. Game Theory and Equilibrium (Von Neumann's Minimax Theorem):**

    * In a zero-sum two-player game, Player 1 seeks to maximize their guaranteed payoff $\max_x \min_y x'Ay$, while Player 2 seeks to minimize Player 1's payoff $\min_y \max_x x'Ay$.
    * Strong Duality proves that $\max \min = \min \max$, establishing the existence of a unique game value and Nash equilibrium.

    ---

    ### 2.2 Primal-Dual Formulations & The Universal Translation Lexicon

    How do we systematically write down the dual of *any* linear program, regardless of its mixture of $\ge, \le, =$ constraints and non-negative, non-positive, or free variables?

    The universal primal-dual translation rules are summarized below:

    #### **Universal Primal-Dual Translation Lexicon**

    | **PRIMAL (Minimize)** | $\longleftrightarrow$ | **DUAL (Maximize)** |
    | :---: | :---: | :---: |
    | **Constraints ($i = 1, \dots, m$)** | | **Variables ($p_1, \dots, p_m$)** |
    | $\ge b_i$ (surplus constraint) | $\longleftrightarrow$ | $p_i \ge 0$ (nonnegative price) |
    | $\le b_i$ (capacity constraint) | $\longleftrightarrow$ | $p_i \le 0$ (nonpositive price) |
    | $= b_i$ (equality constraint) | $\longleftrightarrow$ | $p_i \text{ is free (unrestricted in sign)}$ |
    | **Variables ($x_1, \dots, x_n$)** | | **Constraints ($j = 1, \dots, n$)** |
    | $x_j \ge 0$ | $\longleftrightarrow$ | $\sum_{i=1}^m a_{ij} p_i \le c_j$ |
    | $x_j \le 0$ | $\longleftrightarrow$ | $\sum_{i=1}^m a_{ij} p_i \ge c_j$ |
    | $x_j \text{ is free}$ | $\longleftrightarrow$ | $\sum_{i=1}^m a_{ij} p_i = c_j$ |

    > **Memory Rule (Minimization Primal):**
    >
    > * An equality constraint ($=$) gives a **free** dual variable (you have no slack to play with).
    > * A standard $\ge$ inequality gives a **nonnegative** dual variable ($p_i \ge 0$).
    > * A standard $x_j \ge 0$ variable gives an upper bound on dual pricing: $\sum_i a_{ij} p_i \le c_j$ (the imputed value cannot exceed direct cost).

    #### **Two Canonical Forms in Matrix Notation:**

    **1. Standard Form Primal-Dual Pair:**

    $$
    \begin{aligned}
    \text{(Primal)} \quad & \min \; c'x \quad \text{s.t.} \quad Ax = b,\; x \ge 0 \\
    \text{(Dual)} \quad & \max \; p'b \quad \text{s.t.} \quad p'A \le c' \quad (p \text{ free})
    \end{aligned}
    $$

    **2. Symmetric Form Primal-Dual Pair:**

    $$
    \begin{aligned}
    \text{(Primal)} \quad & \min \; c'x \quad \text{s.t.} \quad Ax \ge b,\; x \ge 0 \\
    \text{(Dual)} \quad & \max \; p'b \quad \text{s.t.} \quad p'A \le c',\; p \ge 0
    \end{aligned}
    $$

    ---

    ### 2.3 The Core Theorems & Proofs

    #### **Theorem 4.1 (Weak Duality Theorem)**
    If $x$ is a feasible solution to the primal problem ($\min c'x, Ax = b, x \ge 0$) and $p$ is a feasible solution to the dual problem ($\max p'b, p'A \le c'$), then:

    $$
    \boxed{p'b \le c'x}
    $$

    **Proof *(One-line elegance)*:**

    $$
    p'b = p'(Ax) = (p'A)x \le c'x
    $$

    The last inequality holds because $p'A \le c'$ (dual feasibility) and $x \ge 0$ (primal feasibility, so multiplying preserves inequalities). $\blacksquare$

    **Immediate Corollaries:**

    1. If $x$ is primal feasible and $p$ is dual feasible with **$c'x = p'b$**, then **$x$ is primal optimal and $p$ is dual optimal**.
    2. If the primal objective is unbounded below ($v^* = -\infty$), the dual **must be infeasible**.
    3. If the dual objective is unbounded above ($+\infty$), the primal **must be infeasible**.

    ---

    #### **Theorem 4.2 (Strong Duality Theorem)**
    If a linear programming problem has an optimal solution $x^*$, then the dual problem also has an optimal solution $p^*$, and their optimal objective values are equal:

    $$
    \boxed{c'x^* = p^{*\top} b}
    $$

    **Constructive Proof via the Simplex Method:**

    1. Let $x^*$ be an optimal basic feasible solution obtained by the Simplex method with optimal basis matrix $B$.
    2. Define the candidate dual vector:

    $$
    p^* \equiv (c_B' B^{-1})' \iff p^{*\top} = c_B' B^{-1}
    $$

    3. Check **Dual Feasibility**: By Simplex optimality (Theorem 3.1), all reduced costs at $x^*$ are nonnegative:

    $$
    \bar{c}' = c' - c_B' B^{-1} A = c' - p^{*\top} A \ge 0 \iff p^{*\top} A \le c'
    $$

    Thus $p^*$ satisfies every dual constraint—$p^*$ is dual feasible!

    4. Check **Objective Value Match**:

    $$
    p^{*\top} b = c_B' B^{-1} b = c_B' x_B^* = c'x^*
    $$

    (since nonbasic variables in $x^*$ are zero).

    5. By Corollary 1 to Weak Duality, since $p^*$ is dual feasible and $p^{*\top}b = c'x^*$, **$p^*$ is an optimal solution to the dual**. $\blacksquare$

    #### **The 4-Case Duality Outcome Matrix:**
    Can both primal and dual be infeasible simultaneously? **Yes!**

    | | **Dual: Finite Optimum** | **Dual: Unbounded ($+\infty$)** | **Dual: Infeasible** |
    | :--- | :---: | :---: | :---: |
    | **Primal: Finite Optimum** | **Possible** ($c'x^* = p^{*\top}b$) | Impossible (Weak Duality) | Impossible (Strong Duality) |
    | **Primal: Unbounded ($-\infty$)** | Impossible (Weak Duality) | Impossible (Weak Duality) | **Possible** |
    | **Primal: Infeasible** | Impossible (Strong Duality) | **Possible** | **Possible** |

    ---

    ### 2.4 The Complementary Slackness Theorem

    Complementary slackness is the mathematical statement of **economic efficiency**: a resource with leftover slack has zero marginal value, and an activity with positive reduced cost will not be produced.

    #### **Theorem 4.3 (Complementary Slackness)**
    Let $x$ be primal feasible and $p$ be dual feasible. Then $x$ and $p$ are **simultaneously optimal** if and only if:

    $$
    \boxed{p_i \cdot (a_i'x - b_i) = 0 \quad \text{for all } i = 1, \dots, m} \qquad \text{(Primal Complementarity)}
    $$

    $$
    \boxed{(c_j - p'A_j) \cdot x_j = \bar{c}_j \cdot x_j = 0 \quad \text{for all } j = 1, \dots, n} \qquad \text{(Dual Complementarity)}
    $$

    **Proof:**

    Subtract the dual objective from the primal objective:

    $$
    c'x - p'b = c'x - p'(Ax - s) = (c' - p'A)x + p's = \sum_{j=1}^n (c_j - p'A_j) x_j + \sum_{i=1}^m p_i s_i
    $$

    where $s = Ax - b \ge 0$ is the surplus/slack. Since $x \ge 0$, $(c' - p'A) \ge 0$, $p \ge 0$, and $s \ge 0$, **every single term in both sums is nonnegative ($\ge 0$)**!
    The sum equals zero ($c'x = p'b \iff \text{Strong Duality}$) **if and only if every individual term vanishes**:

    1. For each $i$: either $a_i'x = b_i$ (constraint is binding / tight), or $p_i = 0$ (shadow price is zero).
    2. For each $j$: either $x_j = 0$ (activity is inactive), or $p'A_j = c_j$ (reduced cost $\bar{c}_j = 0$, marginal value equals marginal cost). $\blacksquare$

    ---

    ### 2.5 Farkas' Lemma & Theorems of the Alternative

    Can we prove whether a system of linear inequalities has a solution without running an algorithm?

    #### **Theorem 4.4 (Farkas' Lemma)**
    Let $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$. Exactly **one** of the following two systems has a solution:

    1. **System 1:** $Ax = b$ and $x \ge 0$ *(primal feasibility: $b$ lies in the cone generated by columns of $A$)*
    2. **System 2:** $p'A \le 0$ and $p'b > 0$ *(dual certificate: a separating hyperplane $p$ separating $b$ from $\operatorname{cone}(A)$)*

    * **Geometric Meaning:** If $b$ is outside the cone of columns of $A$, there exists a hyperplane with normal $p$ such that the entire cone lies on one side ($p'A_j \le 0$ for all $j$) while $b$ lies strictly on the other side ($p'b > 0$).
    """)
    return


@app.cell
def sec10_ui(mo):
    delta_b1_slider = mo.ui.slider(
        start=-3.0, stop=4.0, step=0.5, value=0.0,
        label=r"Perturb Resource 1 RHS ($b_1 = 8 + \Delta b_1$) to observe Marginal Shadow Price $p_1^* = \partial v^* / \partial b_1$:"
    )
    delta_b1_slider
    return (delta_b1_slider,)


@app.cell
def sec10_code(delta_b1_slider, linprog, mo, np, pl, plt):
    # Interactive Duality Laboratory: Shadow Prices, Complementary Slackness & Dual Simplex
    # Primal Problem (Resource Allocation Example):
    # max 3*x1 + 2*x2 + 4*x3  <=>  min -3*x1 - 2*x2 - 4*x3
    # s.t.
    # (1)  x1 +  x2 + 2*x3 <= 8 + \Delta b1   (Resource 1 capacity)
    # (2)        x2 + 6*x3 <= 12              (Resource 2 capacity)
    # (3)  x1              <= 4               (Resource 3 capacity)
    # (4)        x2        <= 6               (Resource 4 capacity)
    # x1, x2, x3 >= 0

    _d_b1 = delta_b1_slider.value
    _b_base = np.array([8.0, 12.0, 4.0, 6.0])
    _b_perturbed = np.array([8.0 + _d_b1, 12.0, 4.0, 6.0])
    _c_obj = np.array([-3.0, -2.0, -4.0])
    _A_sub = np.array([
        [1.0, 1.0, 2.0],
        [0.0, 1.0, 6.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    # Solve Primal & Extract Dual Variables (Shadow Prices) via SciPy HiGHS
    _res_p = linprog(_c_obj, A_ub=_A_sub, b_ub=_b_perturbed, bounds=(0.0, None), method="highs")
    primal_x_val = _res_p.x.copy()
    primal_rc_val = _res_p.lower.marginals.copy()
    primal_obj_val = float(_res_p.fun)
    dual_pi_val = _res_p.ineqlin.marginals.copy()
    primal_slack_val = _res_p.ineqlin.residual.copy()

    # Solve explicit DUAL LP to verify Strong Duality:
    # Let y = -p >= 0 represent the dual resource valuations
    _res_d = linprog(_b_perturbed, A_ub=-_A_sub.T, b_ub=_c_obj, bounds=(0.0, None), method="highs")
    explicit_dual_y = _res_d.x.copy()
    explicit_dual_obj = -float(_res_d.fun)

    # Sensitivity curve across delta_b1 range [-3.0, +4.0]
    _db_vals = np.linspace(-3.0, 4.0, 60)
    _obj_curve = []
    for _db in _db_vals:
        _b_sw = np.array([8.0 + _db, 12.0, 4.0, 6.0])
        _res_sw = linprog(_c_obj, A_ub=_A_sub, b_ub=_b_sw, bounds=(0.0, None), method="highs")
        _obj_curve.append(-float(_res_sw.fun))  # Profit = -cost

    fig_dual, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=110)

    # Left: Profit vs. b1 capacity showing piecewise linear concave curve & shadow price slope
    ax_d1.plot(_db_vals + 8.0, _obj_curve, color="#2563eb", lw=2.5, label=r"Max Profit $v^*(b_1)$")
    _curr_profit = -primal_obj_val
    ax_d1.scatter(8.0 + _d_b1, _curr_profit, color="#dc2626", s=110, edgecolor="black", zorder=6,
                  label=rf"Current $b_1 = {8.0+_d_b1:.1f}$ (Profit = ${_curr_profit:.2f}$)")

    # Draw tangent line with slope = shadow price (-dual_pi_val[0])
    _slope = -dual_pi_val[0]
    _tan_x = np.linspace(8.0 + _d_b1 - 1.5, 8.0 + _d_b1 + 1.5, 50)
    _tan_y = _curr_profit + _slope * (_tan_x - (8.0 + _d_b1))
    ax_d1.plot(_tan_x, _tan_y, "r--", lw=2.0, label=rf"Tangent Slope = Shadow Price $p_1^* = ${_slope:.2f}/unit")

    ax_d1.set_xlabel("Resource 1 Capacity $b_1$")
    ax_d1.set_ylabel("Optimal Profit ($-v^*$)")
    ax_d1.set_title(r"Shadow Price as Marginal Profit: $\partial v^* / \partial b_1 = p_1^*$")
    ax_d1.legend(loc="lower right", fontsize=8)
    ax_d1.grid(True, alpha=0.3)

    # Right: Dual Valuation of Resources (Bar chart of shadow prices)
    _res_names = ["Res 1\n(Mix)", "Res 2\n(Machine)", "Res 3\n(Supply)", "Res 4\n(Labor)"]
    _bars = ax_d2.bar(_res_names, -dual_pi_val, color=["#2563eb", "#7c3aed", "#059669", "#d97706"], width=0.55)
    ax_d2.set_ylabel("Shadow Price ($/unit)")
    ax_d2.set_title("Optimal Resource Shadow Prices $p_i^*$")
    for _bar, _val in zip(_bars, -dual_pi_val):
        ax_d2.text(_bar.get_x() + _bar.get_width()/2.0, _bar.get_height() + 0.05, f"${_val:.2f}",
                   ha="center", fontsize=9, weight="bold")
    ax_d2.set_ylim(0, max(-dual_pi_val.min(), -dual_pi_val.max()) + 1.2)
    ax_d2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Build Complementary Slackness Verification Tables (Polars)
    _df_comp_primal = pl.DataFrame({
        "Constraint i": ["1: Res 1", "2: Res 2", "3: Res 3", "4: Res 4"],
        "RHS b_i": np.round(_b_perturbed, 2),
        "LHS A_i x": np.round(_b_perturbed - primal_slack_val, 2),
        "Slack s_i": np.round(primal_slack_val, 4),
        "Shadow Price p_i*": np.round(-dual_pi_val, 4),
        "Product s_i * p_i*": np.round(primal_slack_val * (-dual_pi_val), 6),
        "Complementary?": ["Tight: s_i=0" if abs(s) < 1e-6 else "Zero Price: p_i=0" for s in primal_slack_val],
    })

    _df_comp_dual = pl.DataFrame({
        "Variable j": ["x_1", "x_2", "x_3"],
        "Optimal Level x_j*": np.round(primal_x_val, 4),
        "Reduced Cost c_j": np.round(primal_rc_val, 4),
        "Product x_j* * c_j": np.round(primal_x_val * primal_rc_val, 6),
        "Complementary?": ["x_j > 0 => c_j = 0" if x > 1e-6 else "x_j = 0 => c_j >= 0" for x in primal_x_val],
    })

    _summary_str = (
        f"### Strong Duality & Economic Verification\n"
        f"- **Primal Optimal Value ($c'x^*$):** **{primal_obj_val:.4f}** (Max Profit = **{-primal_obj_val:.4f}**)\n"
        f"- **Dual Optimal Value ($p' b$):** **{explicit_dual_obj:.4f}**\n"
        f"- **Duality Gap:** $|c'x^* - p' b| =$ **{abs(primal_obj_val - explicit_dual_obj):.2e}** (Zero Gap ✓)\n"
        f"- **Active Shadow Price for Resource 1:** **${-dual_pi_val[0]:.2f} / unit**"
    )

    mo.vstack([
        fig_dual,
        mo.hstack([
            mo.vstack([
                mo.md(_summary_str),
                mo.md("#### 1. Primal Complementary Slackness ($s_i \\cdot p_i^* = 0$):"),
                _df_comp_primal,
            ]),
            mo.vstack([
                mo.md("#### 2. Dual Complementary Slackness ($x_j^* \\cdot \\bar{c}_j = 0$):"),
                _df_comp_dual,
                mo.md(
                    r"> **Economic Law:** A resource with positive leftover slack ($s_i > 0$) has a shadow price of **exactly \$0**! "
                    "You only pay a positive shadow price ($p_i^* > 0$) for bottlenecks that are 100% utilized ($s_i = 0$)."
                )
            ]),
        ], widths=[0.55, 0.45])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Master Synthesis: Modules 01 & 02 Unified Architecture
    """)
    return


@app.cell(hide_code=True)
def sec8_summary(mo):
    mo.md(r"""
    | Foundation / Module | Core Mathematical Question | Geometric Picture ($\mathbb{R}^n$) | Algebraic / Simplex Picture ($Ax=b, x\ge 0$) | Key Theorems & Results |
    | :--- | :--- | :--- | :--- | :--- |
    | **1. Polyhedra & Convex Sets** | What is the shape of feasible sets? | Intersection of $m$ halfspaces; all line segments $[x,y]$ stay inside $P$. | Convex combination $\lambda x + (1-\lambda)y$ preserves $Ax \ge b$. | **Theorem 2.1**; *Every local minimum is a global minimum*. |
    | **2. Extreme Points, Vertices & BFS** | What is a "corner point"? | **Extreme Point:** Not a strict midpoint. **Vertex:** Unique minimizer of some $c'x$. | **BFS:** Exactly $n$ linearly independent active constraints ($a_i'x = b_i$). | **Theorem 2.3** ($\text{Vertex} \iff \text{Extreme Point} \iff \text{BFS}$); at most $\binom{m}{n}$ corners. |
    | **3. Polyhedra in Standard Form** | How do computers represent polyhedra? | Synthesizing vector $b$ inside the positive cone $\operatorname{cone}(A_1, \dots, A_n)$. | Nonbasic $x_N = 0$ ($n-m$ zeros); Basic $x_B = B^{-1}b \ge 0$ ($\det(B) \ne 0$). | **Theorem 2.4**; Full row rank $\operatorname{rank}(A)=m$ is without loss of generality (**Theorem 2.5**). |
    | **4. Degeneracy** | What happens when $>n$ walls meet at a point? | Overdetermined corner: 3+ lines in 2D, 4+ planes in 3D. | $> n - m$ zeros; basic variable $(B^{-1}b)_i = 0$; multiple bases $B_1, B_2$ yield same $x$. | **Defs 2.10 & 2.11**; RHS perturbation $b + \epsilon$ splits degenerate corners into nondegenerate ones. |
    | **5. Existence of Extreme Points** | When does a polyhedron have corners? | Polyhedron contains no infinite two-way line $x + \lambda d$ ($\forall \lambda \in \mathbb{R}$). | Constraint normals have full rank: $\operatorname{rank}(A) = n$ ($\mathcal{N}(A) = \{0\}$). | **Theorem 2.6** (slide-to-corner algorithm); always holds for bounded & standard-form $P$. |
    | **6. Optimality of Extreme Points** | Why only check corners? | Optimal face $Q = \{x \in P \mid c'x = v^*\}$ has corners, and every corner of $Q$ is a corner of $P$. | If finite optimal cost exists, at least one BFS $x_B = B^{-1}b \ge 0$ achieves $v^*$. | **Theorems 2.7 & 2.8**; Fundamental Theorem of Linear Programming. |
    | **7. Representation of Bounded Polyhedra\*** | External ($\mathcal{H}$) vs. Internal ($\mathcal{V}$) view | Every polytope is the rubber-band convex hull of its extreme points. | $x = \sum_{i=1}^K \mu_i x^i$ ($\mu_i \ge 0, \sum \mu_i = 1$). Foundation of Dantzig-Wolfe decomposition. | **Theorem 2.9** (Constructive ray-shooting induction on dimension). |
    | **8. The Simplex Method** | How do we find the optimal corner in practice? | Walking along 1D edges from corner to adjacent corner in cost-reducing directions. | Pivot step: choose entering variable with $\bar{c}_j < 0$, ratio test $\theta^* = \min_{u_i>0} x_{B(i)}/u_i$ to find leaving variable. | **Theorem 3.1** (Optimality if $\bar{c} \ge 0$); **Bland's Rule** (no cycling); **Two-Phase Simplex**. |
    | **9. Duality Theory** | Why do we need the dual problem? | Primal allocates resources; Dual values resources via separating hyperplanes. | Primal: $\min c'x, Ax=b, x\ge 0$. Dual: $\max p'b, p'A \le c'$. Duality gap closes to 0 at optimality. | **Theorems 4.1 & 4.2** (Weak & Strong Duality); **Theorem 4.3** (Complementary Slackness); **Farkas' Lemma**. |
    | **10. Economic & Algorithmic Applications of Duality** | In which cases do we use duality? | Resource valuation & separating hyperplanes for large-scale decomposition. | Shadow prices ($\frac{\partial v^*}{\partial b_i} = p_i^*$), Dual Simplex pivots when $m \gg n$ or branching. | MIP Branch-and-Bound re-optimization, Benders / L-shaped cuts, Column Generation. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. **Bertsimas, D., & Tsitsiklis, J. N.** (1997). *Introduction to Linear Optimization*. Athena Scientific.
    2. **Dantzig, G. B.** (1963). *Linear Programming and Extensions*. Princeton University Press.
    3. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
    """)
    return


if __name__ == "__main__":
    app.run()
