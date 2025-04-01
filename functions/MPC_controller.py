import cvxpy as cp  # For MPC optimization
def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=None):
    """
    Solve the MPC problem over a horizon N and adjust the altitude (z) control with a PID controller.

    The optimization problem is:

      Minimize   sum_{k=0}^{N-1} [ (x[k]-ref)'Q(x[k]-ref) + u[k]'Ru[k] ] + (x[N]-ref)'Q(x[N]-ref)
      Subject to: x[k+1] = A_d x[k] + B_d u[k]   for k=0,...,N-1
                  u_min <= u[k] <= u_max         for k=0,...,N-1
                  x[0] = x0

    Inputs:
      x0          : Current state (12-dimensional vector).
      A_d, B_d    : Discrete-time system matrices.
      ref         : Reference state (12-dimensional vector).
      N           : Prediction horizon.
      Q           : State cost matrix (12x12).
      R           : Input cost matrix (4x4).
      u_min, u_max: Lower and upper bounds on control input (4-dimensional vectors).
      pid_controller: (Optional) An instance of PIDController for z altitude error.

    Returns:
      u_opt : The optimal control input for the current step (4-dimensional vector), with PID adjustment if applicable.
      x_pred: The predicted state trajectory (12 x (N+1) array).
    """
    n = A_d.shape[0]      # 12 states
    m_inputs = B_d.shape[1]  # 4 control inputs

    # Define optimization variables:
    x = cp.Variable((n, N+1))
    u = cp.Variable((m_inputs, N))
    cost = 0
    constraints = []
    
    constraints.append(x[:, 0] == x0)
    for k in range(N):
        cost += cp.quad_form(x[:, k] - ref, Q) + cp.quad_form(u[:, k], R)
        constraints.append(x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k])
        constraints.append(u[:, k] >= u_min)
        constraints.append(u[:, k] <= u_max)
 

    cost += cp.quad_form(x[:, N] - ref, Q)
    
    # Solve the optimization problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception("MPC optimization did not solve to optimality")
    
    u_opt = u[:, 0].value
    x_pred = x.value

    # Apply low-level PID correction for altitude error if provided.
    # (Assumes the z-component of the state is at index 2 and the z control is at u[0].)
    if pid_controller is not None:
        error_z = ref[2] - x0[2]  # altitude error
        pid_output = pid_controller.update(error_z)
        if m_inputs > 0:
            u_opt[0] += pid_output
            #u_opt[0] = min(u_opt[0], u_max[0])
        else:
            print("Warning: No control input available for z correction.")




    return u_opt, x_pred