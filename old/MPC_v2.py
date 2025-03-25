import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random
import noise

#------------------------------------------------------------------
# A simple PID controller class for altitude correction.
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

#------------------------------------------------------------------
# Provided MPC controller function with PID correction on the z channel.
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

#------------------------------------------------------------------
# Main code: Connect to CoppeliaSim and run the MPC loop with PID altitude correction.
client = RemoteAPIClient() 
sim = client.require('sim')

# Get handles for the drone, target, and propellers.
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
propellerHandle = [None] * 4
jointHandle = [None] * 4
sim.setObjectPosition(targetHandle, -1, [0, 0, 2])

for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')

# Set simulation to run in stepping mode.
sim.setStepping(True)
sim.startSimulation()

#------------------------------------------------------------------
# MPC and model parameters
dt = 0.05             # Time step [s]
N = 10                # Prediction horizon (number of steps)
m_drone = 0.52         # Drone mass [kg] (ensure this matches your simulation)
g = 9.81              # Gravity [m/s^2]

# State vector: [x, y, z, vx, vy, vz]
n_states = 6
n_inputs = 3        # Control inputs: desired accelerations in x, y, and z

# Discrete-time dynamics for a double integrator model.
A_d = np.array([[1, 0, 0, dt, 0,  0],
                [0, 1, 0, 0,  dt, 0],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0],
                [0, 0, 0, 0,  1,  0],
                [0, 0, 0, 0,  0,  1]])
B_d = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt]])

# Cost matrices (tune these weights for desired performance).
Q = np.diag([10, 10, 50, 1, 1, 1])
R = np.diag([0.1, 0.1, 0.1])

# Define control input bounds (acceleration limits in m/s^2).
u_min = np.array([-2, -2, -2])
u_max = np.array([2, 2, 2])

# Create an instance of the PID controller for altitude (z-axis).
pid_z = PIDController(kp=0.5, ki=0.1, kd=0.05, dt=dt)

#------------------------------------------------------------------
# Simulation loop
while (t := sim.getSimulationTime()) < 500:
    print(f"Simulation time: {t:.2f} s")
    
    # --- State Estimation ---
    # Get current drone position and velocity.
    pos = sim.getObjectPosition(droneHandle, -1)  # World-frame [x, y, z]
    linVel, _ = sim.getObjectVelocity(droneHandle)  # [vx, vy, vz]
    x0 = np.array([pos[0], pos[1], pos[2],
                   linVel[0], linVel[1], linVel[2]])
    x0 = noise.add_measurement_noise(x0, 0.01) # Simulate sensor noise
    
    # --- Define the Reference State ---
    # We want the drone to reach the target position with zero velocities.
    targetPos = sim.getObjectPosition(targetHandle, -1)
    ref = np.array([targetPos[0], targetPos[1], targetPos[2], 0, 0, 0])
    
    # --- Solve the MPC Problem with PID Correction on Altitude ---
    try:
        u_opt, x_pred = mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=pid_z)
    except Exception as e:
        print("MPC error:", e)
        u_opt = np.zeros(n_inputs)
    
    # --- Convert the Control Input to a Force Command ---
    # The optimal control input u_opt represents a desired acceleration.
    # Compute the desired net force using F = m*(acceleration + gravity compensation).
    F_des = m_drone * (u_opt + np.array([0, 0, g]))
    
    # --- Force Distribution ---
    # For this example, we distribute the net force equally to each of the four propellers.
    force_per_prop = F_des / 4.0
    
    # --- Transform Force from Body to World Frame (if necessary) ---
    # Typically, forces are applied in the drone's body frame.
    # Get the drone’s transformation matrix and zero out its translation.
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0
    world_force = sim.multiplyVector(bodyMatrix, force_per_prop.tolist())
    
    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], world_force, [0, 0, 0])
    
    # Step the simulation
    sim.step()

sim.stopSimulation()
