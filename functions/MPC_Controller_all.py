import cvxpy as cp  # For MPC optimization
import math

def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, 
                   pid_controller_z=None, pid_controller_x=None, pid_controller_y=None):
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
        constraints.append(x[6, k] >= -0.5)  # Roll lower bound
        constraints.append(x[6, k] <= 0.5)   # Roll upper bound
        constraints.append(x[7, k] >= -0.5)  # Pitch lower bound
        constraints.append(x[7, k] <= 0.5)   # Pitch upper bound
 
    cost += cp.quad_form(x[:, N] - ref, Q)
    
    # Solve the optimization problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception("MPC optimization did not solve to optimality")
    
    u_opt = u[:, 0].value
    x_pred = x.value

    # --- Altitude (z-axis) Correction ---
    if pid_controller_z is not None:
        error_z = ref[2] - x0[2]  # altitude error in world frame
        pid_output_z = pid_controller_z.update(error_z)
        u_opt[0] += pid_output_z

    # --- Lateral (x and y) Correction in Body Frame ---
    # Convert world frame lateral errors to the body frame using the current yaw.
    if pid_controller_x is not None or pid_controller_y is not None:
        yaw = x0[8]  # assuming yaw is at index 8
        # Compute world frame errors.
        err_world_x = ref[0] - x0[0]
        err_world_y = ref[1] - x0[1]
        # Transform errors into the body frame.
        error_body_x = math.cos(yaw) * err_world_x + math.sin(yaw) * err_world_y
        error_body_y = -math.sin(yaw) * err_world_x + math.cos(yaw) * err_world_y

    # Use the body frame errors to adjust pitch (affecting x motion) and roll (affecting y motion).
    if pid_controller_x is not None:
        pid_output_x = pid_controller_x.update(error_body_x)
        u_opt[2] += pid_output_x  # pitch command adjustment
        print(err_world_x,error_body_x)

    if pid_controller_y is not None:
        pid_output_y = pid_controller_y.update(error_body_y)
        u_opt[1] -= pid_output_y  # roll command adjustment (note the sign inversion)
    
    return u_opt, x_pred
