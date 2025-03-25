import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random

#------------------------------------------------------------------
# PID Controller for altitude (z) correction.
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
# Modified MPC controller function.
def mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=None):
    """
    Solve the MPC problem over a horizon N and adjust the thrust control with a PID controller.
    
    The optimization problem is:
    
      Minimize   sum_{k=0}^{N-1} [ (x[k]-ref)'Q(x[k]-ref) + u[k]'R u[k] ] + (x[N]-ref)'Q(x[N]-ref)
      Subject to: x[k+1] = A_d x[k] + B_d u[k]   for k=0,...,N-1
                  u_min <= u[k] <= u_max         for k=0,...,N-1
                  x[0] = x0
    
    Inputs:
      x0   : Current state (12-dimensional vector).
      A_d, B_d : Discrete–time system matrices (A_d: 12×12, B_d: 12×4).
      ref  : Reference state (12-dimensional vector). For instance, [x_ref, y_ref, z_ref, 0,0,0,0,0,0,0,0,0].
      N    : Prediction horizon (number of steps).
      Q    : State error cost matrix (12×12).
      R    : Input cost matrix (4×4).
      u_min, u_max : Lower and upper bounds on control input (4-dimensional vectors).
      pid_controller: An instance of PIDController for altitude (z) error (optional).
                      If provided, its output is added to the thrust (u[0]).
    
    Returns:
      u_opt: The optimal control input for the current step (4-dimensional vector), with PID adjustment if applicable.
      x_pred: The predicted state trajectory (12 x (N+1) array).
    """
    n = A_d.shape[0]  # n = 12
    m_in = B_d.shape[1]  # m_in = 4
    
    # Define optimization variables
    x = cp.Variable((n, N+1))
    u = cp.Variable((m_in, N))
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

    # Apply low-level PID controller for altitude error if provided.
    # Here, state index 2 corresponds to altitude and the thrust control input is u[0].
    if pid_controller is not None:
        error_z = ref[2] - x0[2]  # altitude error
        pid_output = pid_controller.update(error_z)
        # Add the PID correction to the thrust channel (index 0)
        u_opt[0] += pid_output

    return u_opt, x_pred

#------------------------------------------------------------------
# Main: Connect to CoppeliaSim and run the MPC loop.
client = RemoteAPIClient() 
sim = client.require('sim')

# Get handles for the drone, target, and propellers.
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
propellerHandle = [None] * 4
jointHandle = [None] * 4

for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')

# Set simulation to run in stepping mode.
sim.setStepping(True)
sim.startSimulation()

#------------------------------------------------------------------
# Model and MPC parameters
dt = 0.05             # Time step [s]
N = 10                # Prediction horizon (steps)

# Drone physical parameters
m_drone = 0.52         # mass [kg]
I_x = +6.667e-05           # moment of inertia about x (roll)
I_y = +0.00753           # moment of inertia about y (pitch)
I_z = +0.00753           # moment of inertia about z (yaw)
g = 9.81              # gravity [m/s^2]

# Continuous-time state matrix (12×12)
A_cont = np.array([
    [ 0,      0,      0,      1,      0,      0,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      1,      0,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      1,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0, -g,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,    g,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      1,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      1,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      1],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
    [ 0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]
])

# Continuous-time input matrix (12×4)
B_cont = np.zeros((12, 4))
B_cont[5, 0]  = 1.0 / m_drone      # Thrust affects vertical acceleration
B_cont[9, 1]  = 1.0 / I_x          # τ_φ affects roll acceleration
B_cont[10, 2] = 1.0 / I_y          # τ_θ affects pitch acceleration
B_cont[11, 3] = 1.0 / I_z          # τ_ψ affects yaw acceleration

# Discretize using Euler approximation
A_d = np.eye(12) + A_cont * dt
B_d = B_cont * dt

# Cost matrices for MPC (dimensions must match states and inputs)
Q = np.diag([1.0, 1.0, 1.0, # x y z
             0.1, 0.1, 0.1, # vx vy vz
             0.5, 0.5, 0.5, # roll pitch yaw
             0.1, 0.1, 0.1]) # p q r

R = np.diag([0.1, 0.1, 0.1, 0.1])

# Control input bounds (u = [F, τ_φ, τ_θ, τ_ψ])
u_min = np.array([-2, -0.5, -0.5, -0.5])
u_max = np.array([ 2,  0.5,  0.5,  0.5])

# Create a PID controller for altitude correction (applied to thrust channel)
pid_z = PIDController(kp=1.0, ki=0.1, kd=0.05, dt=dt)

#------------------------------------------------------------------
# Simulation loop
while (t := sim.getSimulationTime()) < 500:
    print(f"Simulation time: {t:.2f} s")
    
    # --- State Estimation ---
    # Get drone position and velocity.
    pos = sim.getObjectPosition(droneHandle, -1)   # [x, y, z]
    linVel, _ = sim.getObjectVelocity(droneHandle)   # [vx, vy, vz]
    # For the angular states (roll, pitch, yaw and rates), assume measurements are available.
    # For simplicity, we set them to zero here.
    angles = [0, 0, 0]      # roll, pitch, yaw
    angRates = [0, 0, 0]    # p, q, r
    
    # Construct the 12-dimensional state vector.
    x0 = np.array([pos[0], pos[1], pos[2],
                   linVel[0], linVel[1], linVel[2],
                   angles[0], angles[1], angles[2],
                   angRates[0], angRates[1], angRates[2]])
    
    # --- Define the Reference State ---
    # Desired: target x, y, z with zero velocities and level attitude.
    targetPos = sim.getObjectPosition(targetHandle, -1)
    ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0])
    
    # --- Solve the MPC Problem with PID Correction on Altitude ---
    try:
        u_opt, x_pred = mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=pid_z)
    except Exception as e:
        print("MPC error:", e)
        u_opt = np.zeros(4)
    
    # --- Convert Control Input to Force Command ---
    # u_opt[0] is the thrust adjustment (desired net thrust).
    # Compute the net force vector (include gravity compensation)
    F_des = m_drone * (u_opt[0] + g)
    # The torques are given by u_opt[1:], and in this example we distribute forces equally.
    
    # For this example, we distribute the thrust equally among the four propellers.
    force_per_prop = F_des / 4.0
    # (In a more complete model, you would also allocate the torques to adjust the vehicle's attitude.)
    
    # --- Transform Force from Body to World Frame (if needed) ---
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    # Zero out translation components (indices 3, 7, 11)
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0
    world_force = sim.multiplyVector(bodyMatrix, [force_per_prop, 0, 0])  # Assuming thrust acts along body z-axis
    
    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], world_force, [0, 0, 0])
    
    # Step simulation
    sim.step()

sim.stopSimulation()
