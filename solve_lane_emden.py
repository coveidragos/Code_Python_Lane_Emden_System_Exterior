import math
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. Model functions p, q (radial and non-radial variants)
# ============================================================

def p_radial(r: np.ndarray) -> np.ndarray:
    """
    Radial coefficient p(r) used in Theorem 1 and as base for Theorem 2.

    Here: p(r) = 1 / (1 + r^4).
    """
    return 1.0 / (1.0 + r**4)


def q_radial(r: np.ndarray) -> np.ndarray:
    """
    Radial coefficient q(r) used in Theorem 1 and as base for Theorem 2.

    Here: q(r) = exp(-r).
    """
    return np.exp(-r)


def p_nonradial(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Non-radial coefficient p(r, θ) for Theorem 2 illustration.

    p(r, θ) = p_radial(r) * (1 + 0.5 cos(3θ))
    """
    return p_radial(r) * (1.0 + 0.5 * np.cos(3.0 * theta))


def q_nonradial(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Non-radial coefficient q(r, θ) for Theorem 2 illustration.

    q(r, θ) = q_radial(r) * (1 + 0.5 sin(3θ))
    """
    return q_radial(r) * (1.0 + 0.5 * np.sin(3.0 * theta))


# ============================================================
# 2. Integral solver for the radial supersolution (Theorem 1)
# ============================================================

def solve_lane_emden_integral(
    p_func,
    q_func,
    A: float = math.e,
    c: float = 2.0,
    alpha: float = 0.4,
    beta: float = 0.4,
    R_max: float = 100.0,
    num_points: int = 1000,
    max_iter: int = 1000,
    tol: float = 1e-8,
):
    """
    Solves the Lane–Emden–Fowler system in Liouville (s) variable via
    the integral formulation corresponding to the proof of Theorem 1.

    Returns:
        r  : radial grid (r = exp(s))
        u  : radial profile u(r)
        v  : radial profile v(r)
    """
    print("--- Starting Integral Formulation (Theorem 1 / Supersolution) ---")

    s_min = math.log(A)
    s_max = math.log(R_max)
    s_grid = np.linspace(s_min, s_max, num_points)
    r = np.exp(s_grid)

    # Initial guess: constant equal to the asymptotic value c
    y = np.full_like(s_grid, c)
    z = np.full_like(s_grid, c)

    for it in range(max_iter):
        y_prev = y.copy()
        z_prev = z.copy()

        # First component
        G1 = np.exp(2.0 * s_grid) * p_func(np.exp(s_grid)) * (z_prev ** alpha)
        cum_G1 = cumulative_trapezoid(G1, s_grid, initial=0.0)
        I1 = cum_G1[-1] - cum_G1
        cum_I1 = cumulative_trapezoid(I1, s_grid, initial=0.0)
        J1 = cum_I1[-1] - cum_I1
        y_new = c - J1

        # Second component
        G2 = np.exp(2.0 * s_grid) * q_func(np.exp(s_grid)) * (y_prev ** beta)
        cum_G2 = cumulative_trapezoid(G2, s_grid, initial=0.0)
        I2 = cum_G2[-1] - cum_G2
        cum_I2 = cumulative_trapezoid(I2, s_grid, initial=0.0)
        J2 = cum_I2[-1] - cum_I2
        z_new = c - J2

        # Enforce positivity (consistent with theoretical framework)
        y_new = np.maximum(y_new, 1e-10)
        z_new = np.maximum(z_new, 1e-10)

        diff = np.max(np.abs(y_new - y_prev)) + np.max(np.abs(z_new - z_prev))
        y, z = y_new, z_new

        if diff < tol:
            print(f"[Integral Solver] Converged after {it + 1} iterations. Error: {diff:.2e}")
            break
    else:
        print("[Integral Solver] Warning: Maximum iterations reached without full convergence.")

    # Return u(r) = y(ln r), v(r) = z(ln r)
    return r, y, z


# ============================================================
# 3. 2D polar finite difference solver (Theorem 2)
# ============================================================

def solve_lane_emden_2d_polar(
    p_func_2d,
    q_func_2d,
    R0: float = 1.0,
    R_max: float = 20.0,
    N_r: int = 50,
    N_theta: int = 60,
    c_val: float = 2.0,
    alpha: float = 0.4,
    beta: float = 0.4,
    bc_inner=(0.0, 0.0),
    max_iter: int = 50,
    tol: float = 1e-5,
):
    """
    Solves the Lane–Emden–Fowler system in a 2D exterior domain using
    a finite-difference scheme on a polar grid (r_i, θ_j).

    Boundary conditions:
        u(R0, θ) = u_inner
        v(R0, θ) = v_inner
        u(R_max, θ) = c_val
        v(R_max, θ) = c_val

    Returns:
        R, T  : polar grids (for plotting)
        U     : solution component u(r, θ)
        V     : solution component v(r, θ)
    """
    print("--- Starting 2D Finite Difference (Theorem 2 - Non-Radial Case) ---")

    r = np.linspace(R0, R_max, N_r)
    theta = np.linspace(0.0, 2.0 * np.pi, N_theta, endpoint=False)
    hr = r[1] - r[0]
    ht = theta[1] - theta[0]

    u_inner, v_inner = bc_inner

    # Full polar grid for plotting and coefficient evaluation
    R, T = np.meshgrid(r, theta, indexing="ij")  # shape (N_r, N_theta)

    # Unknowns are at interior radial indices i = 1..N_r-2 (N_r-2 rows),
    # for all angular indices j = 0..N_theta-1.
    n_r_inner = N_r - 2
    n_theta = N_theta
    n_unknowns = n_r_inner * n_theta

    # Precompute radial values at interior nodes
    r_inner = r[1:-1]

    # ------------------------------------------------------------
    # Build sparse Laplace operator in polar coordinates
    # ------------------------------------------------------------
    rows = []
    cols = []
    vals = []

    for i in range(n_r_inner):
        r_i = r_inner[i]
        c_cen = -2.0 / hr**2 - 2.0 / (r_i**2 * ht**2)
        c_up = 1.0 / hr**2 + 1.0 / (2.0 * r_i * hr)
        c_dn = 1.0 / hr**2 - 1.0 / (2.0 * r_i * hr)
        c_th = 1.0 / (r_i**2 * ht**2)

        for j in range(n_theta):
            row_idx = i * n_theta + j

            # center
            rows.append(row_idx)
            cols.append(row_idx)
            vals.append(c_cen)

            # angular neighbors (periodic)
            j_next = (j + 1) % n_theta
            j_prev = (j - 1 + n_theta) % n_theta

            rows.append(row_idx)
            cols.append(i * n_theta + j_next)
            vals.append(c_th)

            rows.append(row_idx)
            cols.append(i * n_theta + j_prev)
            vals.append(c_th)

            # radial neighbors inside interior
            if i < n_r_inner - 1:
                rows.append(row_idx)
                cols.append((i + 1) * n_theta + j)
                vals.append(c_up)

            if i > 0:
                rows.append(row_idx)
                cols.append((i - 1) * n_theta + j)
                vals.append(c_dn)

    L_mat = csr_matrix((vals, (rows, cols)), shape=(n_unknowns, n_unknowns))

    # ------------------------------------------------------------
    # Nonlinear iteration
    # ------------------------------------------------------------
    U_flat = np.full(n_unknowns, c_val * 0.5)
    V_flat = np.full(n_unknowns, c_val * 0.5)

    for it in range(max_iter):
        U_prev = U_flat.copy()
        V_prev = V_flat.copy()

        # Evaluate p, q at interior grid points
        R_u = R[1:-1, :].ravel()
        T_u = T[1:-1, :].ravel()
        P_vals = p_func_2d(R_u, T_u)
        Q_vals = q_func_2d(R_u, T_u)

        # --------------------
        # Solve for U (u component)
        # --------------------
        RHS_U = -P_vals * np.abs(V_prev) ** alpha

        # Inner and outer radial boundary corrections for U
        for j in range(n_theta):
            # i = 0 (inner neighbor)
            r0 = r_inner[0]
            c_dn_0 = 1.0 / hr**2 - 1.0 / (2.0 * r0 * hr)
            idx_0 = 0 * n_theta + j
            RHS_U[idx_0] -= c_dn_0 * u_inner

            # i = n_r_inner-1 (outer neighbor)
            r_last = r_inner[-1]
            c_up_last = 1.0 / hr**2 + 1.0 / (2.0 * r_last * hr)
            idx_last = (n_r_inner - 1) * n_theta + j
            RHS_U[idx_last] -= c_up_last * c_val

        U_sol = spsolve(L_mat, RHS_U)

        # --------------------
        # Solve for V (v component)
        # --------------------
        RHS_V = -Q_vals * np.abs(U_prev) ** beta

        # Inner and outer radial boundary corrections for V
        for j in range(n_theta):
            r0 = r_inner[0]
            c_dn_0 = 1.0 / hr**2 - 1.0 / (2.0 * r0 * hr)
            idx_0 = 0 * n_theta + j
            RHS_V[idx_0] -= c_dn_0 * v_inner

            r_last = r_inner[-1]
            c_up_last = 1.0 / hr**2 + 1.0 / (2.0 * r_last * hr)
            idx_last = (n_r_inner - 1) * n_theta + j
            RHS_V[idx_last] -= c_up_last * c_val

        V_sol = spsolve(L_mat, RHS_V)

        U_flat = np.maximum(U_sol, 0.0)
        V_flat = np.maximum(V_sol, 0.0)

        err = np.max(np.abs(U_flat - U_prev)) + np.max(np.abs(V_flat - V_prev))
        if it % 10 == 0:
            print(f"[2D Solver] Iter {it}: err={err:.2e}")
        if err < tol:
            print(f"[2D Solver] Converged after {it + 1} iterations. err={err:.2e}")
            break

    # ------------------------------------------------------------
    # Reconstruct full grids U(r, θ), V(r, θ) including boundaries
    # ------------------------------------------------------------
    U_full = np.zeros((N_r, N_theta))
    V_full = np.zeros((N_r, N_theta))

    U_full[1:-1, :] = U_flat.reshape((n_r_inner, n_theta))
    V_full[1:-1, :] = V_flat.reshape((n_r_inner, n_theta))

    U_full[0, :] = u_inner
    U_full[-1, :] = c_val
    V_full[0, :] = v_inner
    V_full[-1, :] = c_val

    return R, T, U_full, V_full


# ============================================================
# 4. Plotting utilities
# ============================================================

def plot_theorem1_radial(r, u, v, filename="Theorem1_Supersolution_2D.png"):
    """
    2D plot of the radial supersolution components u(r), v(r) for Theorem 1.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(r, u, "b-", linewidth=2, label="u(r)")
    plt.plot(r, v, "r--", linewidth=2, label="v(r)")
    plt.xlabel("r")
    plt.ylabel("Value")
    plt.title("Theorem 1: Radial Supersolution (u, v)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def plot_bounds_3d(R, T, U_sup, V_sup, filename="Theorem2_Bounds_3D.png"):
    """
    3D plot: subsolution (0) and supersolution surfaces for both components.
    """
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Zero = np.zeros_like(X)

    fig = plt.figure(figsize=(14, 6))

    # u component
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, U_sup, cmap="viridis", alpha=0.6)
    ax1.plot_surface(X, Y, Zero, color="gray", alpha=0.3)
    ax1.set_title("Bounds for u: 0 ≤ u ≤ supersolution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")

    # v component
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, V_sup, cmap="plasma", alpha=0.6)
    ax2.plot_surface(X, Y, Zero, color="gray", alpha=0.3)
    ax2.set_title("Bounds for v: 0 ≤ v ≤ supersolution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("v")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def plot_solution_3d(R, T, U, V, filename="Theorem2_Solution_Components_3D.png"):
    """
    3D plots of the non-radial solution components u(x, y), v(x, y).
    """
    X = R * np.cos(T)
    Y = R * np.sin(T)

    fig = plt.figure(figsize=(14, 6))

    # u component
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, U, cmap="viridis", edgecolor="none")
    ax1.set_title("Theorem 2: Solution component u(x, y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # v component
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, V, cmap="plasma", edgecolor="none")
    ax2.set_title("Theorem 2: Solution component v(x, y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("v")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def plot_comparison_3d(
    R,
    T,
    U_sol,
    V_sol,
    U_sup,
    V_sup,
    filename="Theorem2_Comparison_3D.png",
):
    """
    3D comparison between:
        - subsolution (0),
        - supersolution (radial) and
        - non-radial solution,
    for both components u and v, with high visual contrast.
    """
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Zero = np.zeros_like(X)

    fig = plt.figure(figsize=(14, 6))

    # ------------------------------------
    # Component u
    # ------------------------------------
    ax1 = fig.add_subplot(121, projection="3d")

    # subsolution 0
    ax1.plot_surface(X, Y, Zero, color="gray", alpha=0.25)

    # supersolution: magenta dashed wireframe
    ax1.plot_surface(X, Y, U_sup, color="#FF00FF", alpha=0.10)
    ax1.plot_wireframe(
        X, Y, U_sup, color="#FF00FF", linewidth=1.4, linestyle="--", label="Supersolution"
    )

    # solution: cyan solid wireframe
    ax1.plot_surface(X, Y, U_sol, color="#00FFFF", alpha=0.20)
    ax1.plot_wireframe(
        X, Y, U_sol, color="#00FFFF", linewidth=1.4, linestyle="-", label="Solution u"
    )

    ax1.set_title("Component u: cyan = solution, magenta = supersolution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    ax1.legend()

    # ------------------------------------
    # Component v
    # ------------------------------------
    ax2 = fig.add_subplot(122, projection="3d")

    # subsolution 0
    ax2.plot_surface(X, Y, Zero, color="gray", alpha=0.25)

    # supersolution: red dashed wireframe
    ax2.plot_surface(X, Y, V_sup, color="#FF0000", alpha=0.10)
    ax2.plot_wireframe(
        X, Y, V_sup, color="#FF0000", linewidth=1.4, linestyle="--", label="Supersolution"
    )

    # solution: green solid wireframe
    ax2.plot_surface(X, Y, V_sol, color="#00FF00", alpha=0.20)
    ax2.plot_wireframe(
        X, Y, V_sol, color="#00FF00", linewidth=1.4, linestyle="-", label="Solution v"
    )

    ax2.set_title("Component v: green = solution, red = supersolution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("v")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


def plot_differences_3d(
    R,
    T,
    U_sol,
    V_sol,
    U_sup,
    V_sup,
    filename="Theorem2_Differences_3D.png",
):
    """
    3D plots of the differences U_sup - U_sol and V_sup - V_sol to
    quantify how far the non-radial solution stays below the supersolution.
    """
    X = R * np.cos(T)
    Y = R * np.sin(T)

    DU = U_sup - U_sol
    DV = V_sup - V_sol

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, DU, cmap="viridis", edgecolor="none")
    ax1.set_title("Difference U_sup - U_sol")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("U_sup - U_sol")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, DV, cmap="plasma", edgecolor="none")
    ax2.set_title("Difference V_sup - V_sol")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("V_sup - V_sol")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()


# ============================================================
# 5. Main script (Theorem 1 and Theorem 2 workflow)
# ============================================================

if __name__ == "__main__":
    # Parameters consistent with your paper
    A = 1.0
    c = 2.0
    alpha = 0.4
    beta = 0.4

    # ----------------------------------------
    # Theorem 1: radial supersolution
    # ----------------------------------------
    r_sup, u_sup, v_sup = solve_lane_emden_integral(
        p_radial, q_radial,
        A=A,
        c=c,
        alpha=alpha,
        beta=beta,
        R_max=100.0,
        num_points=1000,
        max_iter=1000,
        tol=1e-8,
    )
    plot_theorem1_radial(r_sup, u_sup, v_sup)

    # Build a 2D radial supersolution grid for bounds plot
    N_grid_r = 50
    N_grid_theta = 60
    r_grid = np.linspace(A, 20.0, N_grid_r)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, N_grid_theta, endpoint=False)
    R_mesh, T_mesh = np.meshgrid(r_grid, theta_grid, indexing="ij")

    u_sup_1d = np.interp(r_grid, r_sup, u_sup)
    v_sup_1d = np.interp(r_grid, r_sup, v_sup)

    U_SUP_MESH = np.tile(u_sup_1d[:, np.newaxis], (1, N_grid_theta))
    V_SUP_MESH = np.tile(v_sup_1d[:, np.newaxis], (1, N_grid_theta))

    plot_bounds_3d(R_mesh, T_mesh, U_SUP_MESH, V_SUP_MESH)

    # ----------------------------------------
    # Theorem 2: non-radial solution via 2D solver
    # ----------------------------------------
    bc_inner_vals = (u_sup[0], v_sup[0])  # match radial supersolution at inner boundary

    print("\n--- Non-Radial Calculation for Theorem 2 ---")
    R_sol, T_sol, U_sol, V_sol = solve_lane_emden_2d_polar(
        p_nonradial,
        q_nonradial,
        R0=A,
        R_max=20.0,
        N_r=50,
        N_theta=60,
        c_val=c,
        alpha=alpha,
        beta=beta,
        bc_inner=bc_inner_vals,
        max_iter=50,
        tol=1e-5,
    )

    plot_solution_3d(R_sol, T_sol, U_sol, V_sol)

    # Interpolate supersolution onto the same (R_sol, T_sol) grid
    u_sup_interp = np.interp(R_sol.ravel(), r_sup, u_sup).reshape(R_sol.shape)
    v_sup_interp = np.interp(R_sol.ravel(), r_sup, v_sup).reshape(R_sol.shape)

    # Print norms of the difference for quantitative comparison
    max_diff_u = np.max(np.abs(U_sol - u_sup_interp))
    max_diff_v = np.max(np.abs(V_sol - v_sup_interp))
    print(f"max |u_sol - u_sup| = {max_diff_u:.3e}")
    print(f"max |v_sol - v_sup| = {max_diff_v:.3e}")

    # Comparison plots
    plot_comparison_3d(R_sol, T_sol, U_sol, V_sol, u_sup_interp, v_sup_interp)
    plot_differences_3d(R_sol, T_sol, U_sol, V_sol, u_sup_interp, v_sup_interp)
