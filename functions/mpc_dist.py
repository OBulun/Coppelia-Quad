import cvxpy as cp
import numpy as np

def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max,
                   pid_controller_z=None, d_hat=None):
    """
    MPC with optional offset-free output bias d_hat (R^3) on position [x,y,z].
    If d_hat is provided, the cost penalizes (S x + d_hat - ref_pos).
    """
    if not hasattr(mpc_controller, "initialized"):
        n = A_d.shape[0]
        m = B_d.shape[1]

        # Decision variables
        x = cp.Variable((n, N+1))
        u = cp.Variable((m, N))

        # Parameters
        x0_par  = cp.Parameter(n)
        ref_par = cp.Parameter(n)
        d_par   = cp.Parameter(3)      # NEW: disturbance on outputs

        # Selector S picks [x,y,z] from state
        S = np.zeros((3, n))
        S[0, 0] = 1.0  # x
        S[1, 1] = 1.0  # y
        S[2, 2] = 1.0  # z

        cost = 0
        constr = [x[:, 0] == x0_par]

        for k in range(N):
            # Split reference into position (first 3) and the rest
            y_ref = ref_par[:3]
            # Output with bias: y = S x + d
            y_k = S @ x[:, k] + d_par

            # Penalize output error with bias (offset-free)
            cost += cp.quad_form(y_k - y_ref, np.diag([Q[0,0], Q[1,1], Q[2,2]]))

            # Penalize the rest of the state as you already do
            rest_err = x[:, k] - ref_par
            # remove the first 3 components so we don't double-count
            rest_err = cp.hstack([0,0,0, rest_err[3:]])
            cost += cp.quad_form(rest_err, Q)

            # Input penalty
            cost += cp.quad_form(u[:, k], R)

            # Dynamics + bounds
            constr += [
                x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k],
                u[:, k] >= u_min,
                u[:, k] <= u_max
            ]

        # Terminal position error with bias
        yN = S @ x[:, N] + d_par
        cost += cp.quad_form(yN - ref_par[:3], np.diag([Q[0,0], Q[1,1], Q[2,2]]))

        prob = cp.Problem(cp.Minimize(cost), constr)

        solve_kwargs = dict(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
        )

        # Stash
        mpc_controller.prob        = prob
        mpc_controller.x0_par      = x0_par
        mpc_controller.ref_par     = ref_par
        mpc_controller.u_var       = u
        mpc_controller.d_par       = d_par
        mpc_controller.initialized = True
        mpc_controller.solve_kwargs = solve_kwargs

    # Update parameters every call
    mpc_controller.x0_par.value  = np.array(x0).flatten()
    mpc_controller.ref_par.value = np.array(ref).flatten()

    # Provide d_hat (or zero if not given)
    if d_hat is None:
        mpc_controller.d_par.value = np.zeros(3)
    else:
        mpc_controller.d_par.value = np.array(d_hat).flatten()

    # Solve
    mpc_controller.prob.solve(**mpc_controller.solve_kwargs)
    if mpc_controller.prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("MPC optimization did not solve to optimality")

    u_opt  = mpc_controller.u_var[:, 0].value
    x_pred = mpc_controller.prob.variables()[0].value  # x.value

    # Keep your altitude PID (optional)
    if pid_controller_z is not None:
        error_z   = ref[2] - x0[2]
        pid_out   = pid_controller_z.update(error_z)
        u_opt[0] += pid_out

    return u_opt, x_pred
