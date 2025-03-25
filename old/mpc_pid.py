#!/usr/bin/env python3
import cvxpy as cp
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, integrator_limit=None):
        """
        Initialize the PID controller.
        
        Parameters:
          Kp, Ki, Kd: PID gains.
          dt: Time step.
          integrator_limit: Optional limit to clamp the integral term.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.integrator_limit = integrator_limit

    def reset(self):
        """Reset the PID controller's internal state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        """
        Update the PID controller.
        
        Parameters:
          error: The current error value.
        
        Returns:
          The PID output.
        """
        self.integral += error * self.dt
        if self.integrator_limit is not None:
            # Clamp the integral term to prevent windup.
            self.integral = np.clip(self.integral, -self.integrator_limit, self.integrator_limit)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=None):
    """
    Solve the MPC problem over a horizon N and adjust the altitude control with a PID controller.
    
    The optimization problem is:
    
      Minimize   sum_{k=0}^{N-1} [ (x[k]-ref)'Q(x[k]-ref) + u[k]'Ru[k] ] + (x[N]-ref)'Q(x[N]-ref)
      Subject to: x[k+1] = A_d x[k] + B_d u[k]   for k=0,...,N-1
                  u_min <= u[k] <= u_max         for k=0,...,N-1
                  x[0] = x0
    
    Inputs:
      x0   : Current state (n-dimensional vector).
      A_d, B_d : Discrete–time system matrices.
      ref  : Reference state (n-dimensional vector). In our case, it includes the desired altitude.
      N    : Prediction horizon (number of steps).
      Q    : State error cost matrix (n×n).
      R    : Input cost matrix (m×m).
      u_min, u_max : Lower and upper bounds on control input (m-dimensional vectors).
      pid_controller: An instance of PIDController for z altitude error (optional).
                      If provided, the PID output will be added to the z component of the MPC control.
    
    Returns:
      u_opt: The optimal control input for the current step (m-dimensional vector), with PID adjustment if applicable.
      x_pred: The predicted state trajectory (n x (N+1) array).
    """
    n = A_d.shape[0]
    m = B_d.shape[1]
    
    # Define optimization variables
    x = cp.Variable((n, N+1))
    u = cp.Variable((m, N))
    cost = 0
    constraints = []
    
    constraints += [x[:, 0] == x0]
    for k in range(N):
        cost += cp.quad_form(x[:, k] - ref, Q) + cp.quad_form(u[:, k], R)
        constraints += [x[:, k+1] == A_d @ x[:, k] + B_d @ u[:, k],
                        u[:, k] >= u_min,
                        u[:, k] <= u_max]
    cost += cp.quad_form(x[:, N] - ref, Q)
    
    # Solve the optimization problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception("MPC optimization did not solve to optimality")
    
    u_opt = u[:, 0].value
    x_pred = x.value

    # Apply low-level PID controller for z altitude error if provided.
    # Here, we assume the third state (index 2) corresponds to the altitude (z) 
    # and the corresponding control input is also at index 2.
    if pid_controller is not None:
        error_z = ref[2] - x0[2]  # Desired altitude minus current altitude.
        pid_output = pid_controller.update(error_z)
        # If the control vector supports a z control input, add the PID correction.
        if m > 2:
            u_opt[2] += pid_output
        else:
            print("Warning: Control input dimension does not include a dedicated z control.")

    return u_opt, x_pred

