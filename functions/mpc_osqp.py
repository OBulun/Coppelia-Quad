import numpy as np
import osqp
import scipy.sparse as sparse

def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=None):
    """
    Solve the MPC problem over a horizon N using a direct QP formulation and OSQP,
    then apply an optional PID correction for the altitude (z) control.

    The original optimization is:
        Minimize   sum_{k=0}^{N-1} [ (x[k]-ref)'Q(x[k]-ref) + u[k]'R u[k] ] + (x[N]-ref)'Q(x[N]-ref)
        Subject to x[k+1] = A_d x[k] + B_d u[k]  for k=0,...,N-1
                   u_min <= u[k] <= u_max        for k=0,...,N-1
                   x[0] = x0

    We condense the dynamics:
        X = A_bar * x0 + B_bar * U,
    where U = [u0, ..., u_{N-1}] is the optimization variable.
    
    The QP in U becomes:
        minimize  1/2 * U' H U + f' U
        subject to u_min_rep <= U <= u_max_rep,
    with
        H = 2*(B_bar^T Q_bar B_bar + R_bar)  and  f = 2*B_bar^T Q_bar (A_bar*x0 - r_vec).

    Inputs:
      x0          : Current state (n-dimensional vector).
      A_d, B_d    : Discrete-time system matrices.
      ref         : Reference state (n-dimensional vector).
      N           : Prediction horizon.
      Q           : State cost matrix.
      R           : Input cost matrix.
      u_min, u_max: Lower and upper bounds on control input (m-dimensional vectors).
      pid_controller: (Optional) An instance of a PIDController for z altitude error.

    Returns:
      u_opt : The optimal control input for the current step (m-dimensional vector),
              adjusted by PID if provided.
      x_pred: The predicted state trajectory (n x (N+1) array).
    """
    n = A_d.shape[0]  # state dimension
    m = B_d.shape[1]  # control input dimension

    # Build the extended state transition matrices A_bar and B_bar.
    A_bar = np.zeros(((N+1)*n, n))
    A_bar[0:n, :] = np.eye(n)
    for i in range(1, N+1):
        A_bar[i*n:(i+1)*n, :] = A_d @ A_bar[(i-1)*n:i*n, :]

    B_bar = np.zeros(((N+1)*n, N*m))
    for i in range(1, N+1):
        for j in range(i):
            A_power = np.linalg.matrix_power(A_d, i-1-j)
            B_bar[i*n:(i+1)*n, j*m:(j+1)*m] = A_power @ B_d

    # Construct block-diagonal matrices for state and input costs.
    Q_blocks = [Q] * (N + 1)
    Q_bar = sparse.block_diag(Q_blocks)
    R_bar = sparse.block_diag([R] * N)

    # Build the stacked reference vector.
    r_vec = np.tile(ref, (N+1, 1)).reshape(((N+1)*n, 1))
    g = A_bar @ x0.reshape(n, 1) - r_vec

    # Convert the necessary matrices to arrays safely.
    B_bar_sparse = sparse.csc_matrix(B_bar)
    H_term = np.asarray(B_bar_sparse.T @ Q_bar @ B_bar_sparse)
    R_term = np.asarray(R_bar)
    H_dense = H_term + R_term

    # OSQP expects the problem in the form: 1/2 * U'H U + f'U.
    H = 2 * H_dense
    f = (2 * np.asarray(B_bar_sparse.T @ (Q_bar @ g))).flatten()

    # Define input constraints by repeating the bounds for each control step.
    u_min_rep = np.tile(u_min, N)
    u_max_rep = np.tile(u_max, N)
    A_constr = sparse.eye(N * m)

    # Set up and solve the QP using OSQP.
    prob = osqp.OSQP()
    prob.setup(P=sparse.csc_matrix(H), q=f, A=A_constr, l=u_min_rep, u=u_max_rep, verbose=False)
    res = prob.solve()

    if res.info.status_val not in [osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")]:
        raise Exception("MPC optimization did not solve to optimality")

    # Retrieve the optimal control sequence.
    U_opt = res.x
    u_opt = U_opt[:m]  # use the first control input

    # Compute the predicted state trajectory.
    x_pred = A_bar @ x0.reshape(n, 1) + B_bar @ U_opt.reshape(-1, 1)
    x_pred = x_pred.reshape((N+1, n)).T

    # Apply PID altitude correction if a PID controller is provided.
    if pid_controller is not None:
        error_z = ref[2] - x0[2]
        pid_output = pid_controller.update(error_z)
        u_opt[0] += pid_output

    return u_opt, x_pred
