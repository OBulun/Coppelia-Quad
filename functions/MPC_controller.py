import cvxpy as cp
import numpy as np

def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller_z=None):

    # On first call, build & stash the problem in the function object
    if not hasattr(mpc_controller, "initialized"):
        n = A_d.shape[0]
        m = B_d.shape[1]

        # Decision variables
        x = cp.Variable((n, N+1))
        u = cp.Variable((m, N))

        # Parameters to update each iteration
        x0_par  = cp.Parameter(n)
        ref_par = cp.Parameter(n)

        cost = 0
        constr = [ x[:, 0] == x0_par ]
        for k in range(N):
            cost   += cp.quad_form(x[:, k]   - ref_par, Q)
            cost   += cp.quad_form(u[:, k],     R)
            constr += [ x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k],
                        u[:, k] >= u_min,
                        u[:, k] <= u_max ]
        cost += cp.quad_form(x[:, N] - ref_par, Q)

        # Build & store
        prob = cp.Problem(cp.Minimize(cost), constr)
        # warm start and choose OSQP
        solve_kwargs = dict(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
        )

        # Attach to the function for reuse
        mpc_controller.prob       = prob
        mpc_controller.x0_par     = x0_par
        mpc_controller.ref_par    = ref_par
        mpc_controller.u_var      = u
        mpc_controller.initialized = True
        mpc_controller.solve_kwargs = solve_kwargs

    # ----- Now on every call -----
    # Update parameters
    mpc_controller.x0_par.value  = np.array(x0).flatten()
    mpc_controller.ref_par.value = np.array(ref).flatten()

    # Solve
    mpc_controller.prob.solve(**mpc_controller.solve_kwargs)
    if mpc_controller.prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("MPC optimization did not solve to optimality")

    # Extract
    u_opt  = mpc_controller.u_var[:, 0].value
    x_pred = mpc_controller.prob.variables()[0].value  # x.value

    # PID altitude correction
    if pid_controller_z is not None:
        error_z   = ref[2] - x0[2]
        pid_out   = pid_controller_z.update(error_z)
        u_opt[0] += pid_out
    return u_opt, x_pred
