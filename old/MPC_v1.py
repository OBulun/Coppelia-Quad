import numpy as np
import cvxpy as cp  # For formulating and solving the MPC optimization problem
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random
import noise

# Connect to the CoppeliaSim server
client = RemoteAPIClient() 
sim = client.require('sim')

# --- Handles and Initialization ---
# Propeller, joint, and force sensor handles (arrays of 4)
propellerHandle = np.array([None, None, None, None])   
jointHandle = np.array([None, None, None, None])
forceSensor = np.array([None, None, None, None])

# Drone and target handles:
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
sim.setObjectPosition(targetHandle, -1, [0.5, 0.6, 1.5])  # Set target position
sim.setObjectPosition(droneHandle, -1, [0, 0, 1])  # Set drone initial position
sim.setObjectOrientation(droneHandle, -1, [0, 0, 0])  # Set drone initial orientation

for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')
    forceSensor[i] = sim.getObject(f'/Quadcopter/propeller[{i}]')

# Set simulation to run in stepping mode
sim.setStepping(True)

# --- MPC Setup ---
# Define simulation and model parameters:
dt = 0.05              # time step [s] (adjust as needed or use sim.getSimulationTimeStep())
N = 10                 # prediction horizon (number of steps)
m = 0.52               # drone mass [kg]
g = 9.81               # gravitational acceleration [m/s²]

# Discrete-time state-space matrices for a double-integrator model (6 states, 3 inputs)
A = np.array([[1, 0, 0, dt, 0,  0],
              [0, 1, 0, 0,  dt, 0],
              [0, 0, 1, 0,  0,  dt],
              [0, 0, 0, 1,  0,  0],
              [0, 0, 0, 0,  1,  0],
              [0, 0, 0, 0,  0,  1]])

B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [dt,0, 0],
              [0, dt,0],
              [0, 0, dt]])

# MPC cost matrices (tune these for performance)
Q = np.diag([0.7, 0.7, 0.7, 1.0, 1.0, 0.1])
R = np.diag([0.1, 0.1, 0.1])

# --- Start Simulation ---
sim.startSimulation()

while (t := sim.getSimulationTime()) < 500:
    print(f'Simulation time: {t:.2f} [s]')
    
    # --- State Estimation ---
    # Get current drone position (world frame)
    pos = sim.getObjectPosition(droneHandle, -1)  # [x, y, z]
    # Get current linear velocity ([vx, vy, vz], ignore angular velocity for this simplified model)
    linVel, _ = sim.getObjectVelocity(droneHandle)
    # Construct the current state vector:
    x0 = np.array([pos[0], pos[1], pos[2],
                   linVel[0], linVel[1], linVel[2]])
    x0 = noise.add_measurement_noise(x0, 0.01) 
    
    # Desired state: target position with zero velocities
    targetPos = sim.getObjectPosition(targetHandle, -1)
    x_ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                      0, 0, 0])
    
    # --- MPC Optimization Problem ---
    # Define optimization variables for states and control inputs over the horizon:
    x_var = cp.Variable((6, N+1))
    u_var = cp.Variable((3, N))
    
    cost = 0
    constraints = []
    constraints += [x_var[:,0] == x0]  # initial condition
    
    for k in range(N):
        # Stage cost: penalize deviation from desired state and control effort
        cost += cp.quad_form(x_var[:,k] - x_ref, Q) + cp.quad_form(u_var[:,k], R)
        # Dynamics constraint: x[k+1] = A*x[k] + B*u[k]
        constraints += [x_var[:,k+1] == A @ x_var[:,k] + B @ u_var[:,k]]
    # Terminal cost (optional):
    cost += cp.quad_form(x_var[:,N] - x_ref, Q)
    
    # Solve the MPC optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)  # You can change the solver if needed
    
    # If the problem is infeasible or u_var is not set, use zero control:
    if u_var[:,0].value is None:
        u_opt = np.zeros(3)
    else:
        u_opt = u_var[:,0].value  # Optimal control input at time k=0
    
    # --- Convert MPC Output to Force Command ---
    # u_opt is the desired acceleration in world frame [ax, ay, az]
    # Compute the desired net force:
    F_des = m * (u_opt + np.array([0, 0, g]))
    # (F_des is in Newtons in the world frame)
    
    # For demonstration, we assume a very simple allocation:
    # Distribute the net force equally to the 4 propellers.
    force_per_prop = F_des /4
    
    # --- Transform Force from Drone Body Frame to World Frame (if needed) ---
    # Typically, forces are applied in the drone's body frame.
    # Here we use the drone's transformation matrix to rotate the force.
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    # Zero out translation components (indices 3, 7, 11 in a 0-indexed 12-element list)
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0
    # Transform the force vector (convert to list for compatibility)
    world_force = sim.multiplyVector(bodyMatrix, force_per_prop.tolist())
    
    # --- Apply Force to Each Propeller ---
    for i in range(4):
        # In this example, we apply the same force vector and no extra torque.
        sim.addForceAndTorque(propellerHandle[i], world_force, [0, 0, 0])
    
    # Step the simulation
    sim.step()

# --- Stop Simulation ---
sim.stopSimulation()
