import numpy as np
import cvxpy as cp
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functions.PID_Controller import PIDController
from functions.save_parameters import save_parameters
from functions.save_logs import save_logs
from functions.mpc_osqp import mpc_controller  # original MPC controller

# =============================================================================
# INITIALIZATION & CONNECTION SETUP
# =============================================================================

client = RemoteAPIClient()
sim = client.require('sim')

# =============================================================================
# SIMULATION PARAMETERS & LOG INITIALIZATION
# =============================================================================

simulation_time = 50  # Total simulation time [s]
dt = sim.getSimulationTimeStep()
N = 30  # MPC prediction horizon

# PID parameters for altitude correction (not used in cascade; set to None below)
kp_z = 0.8
ki_z = 0.1
kd_z = 1.15
pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)

# Data log initialization.
time_log = []
state_log = []
control_log = []
force_log = []
target_log = []

# =============================================================================
# SYSTEM MODEL PARAMETERS & DISCRETIZATION
# =============================================================================

m_drone = 3.61
g = 9.81
bf = 0.0001
l_arm = 0.13

# Get inertia matrix from simulator.
inertiaMatrix, _ = sim.getShapeInertia(sim.getObject('/Quadcopter'))
print(inertiaMatrix)

# Construct continuous-time A and B matrices.
A = np.zeros((12, 12))
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
A[3, 3] = -bf / m_drone
A[3, 7] = g
A[4, 4] = -bf / m_drone
A[4, 6] = -g
A[5, 5] = -bf / m_drone
A[6, 9] = 1
A[7, 10] = 1
A[8, 11] = 1
A[9, 9]  = -bf / inertiaMatrix[0]
A[10,10] = -bf / inertiaMatrix[4]
A[11,11] = -bf / inertiaMatrix[8]

B = np.zeros((12, 4))
B[5, 0] = 1.0 / m_drone
B[9, 1] = 1.0 / inertiaMatrix[0]
B[10,2] = 1.0 / inertiaMatrix[4]
B[11,3] = 1.0 / inertiaMatrix[8]

# Discretize the system.
A_d = np.eye(12) + A * dt
B_d = B * dt

# =============================================================================
# COST MATRICES & CONTROL LIMITS
# =============================================================================

Q = np.diag([5.5, 5.5, 3.0,
             0.05, 0.05, 0.1,
             0.75, 0.75, 0.1,
             0, 0, 0])
R = np.diag([0.1, 0.05, 0.05, 0.1])
u_min = np.array([-m_drone*g, -1.15, -1.15, -1.0])
u_max = np.array([1.0, 1.15, 1.15, 1.0])

save_parameters(Q, R, N, pid_z.get_parameters(), filename="simulation_parameters.txt")

# =============================================================================
# GET HANDLES FOR OBJECTS
# =============================================================================

droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
propellerHandles = [sim.getObject(f'/Quadcopter/propeller[{i}]/respondable') for i in range(4)]
jointHandles = [sim.getObject(f'/Quadcopter/propeller[{i}]/joint') for i in range(4)]

# =============================================================================
# SET INITIAL POSITIONS
# =============================================================================

sim.setObjectPosition(targetHandle, -1, [2.0, 2.0, 2.0])
sim.setObjectPosition(droneHandle, -1, [0.0, 0.0, 1.0])

# =============================================================================
# START SIMULATION
# =============================================================================

sim.setStepping(True)
sim.startSimulation()

# =============================================================================
# CASCADE CONTROL SIMULATION LOOP
# =============================================================================

# Outer-loop gain for desired attitude computation from position error.
Kp_att = 0.1

while (t := sim.getSimulationTime()) < simulation_time:
    print(f"Simulation time: {t:.2f} s")
    
    # --- State Estimation ---
    pos = sim.getObjectPosition(droneHandle, -1)        # [x, y, z]
    linVel, _ = sim.getObjectVelocity(droneHandle)        # [vx, vy, vz]
    ori = sim.getObjectOrientation(droneHandle, -1)       # [roll, pitch, yaw]
    _, angVel = sim.getObjectVelocity(droneHandle)
    
    # Construct the state vector.
    x0 = np.array([pos[0], pos[1], pos[2],
                   linVel[0], linVel[1], linVel[2],
                   ori[0], ori[1], ori[2],
                   angVel[0], angVel[1], angVel[2]])
    time_log.append(t)
    state_log.append(x0)
    
    # --- Outer-Loop: Compute Desired Attitude from Position Error ---
    targetPos = sim.getObjectPosition(targetHandle, -1)
    target_log.append(targetPos)
    pos_error = np.array(targetPos) - np.array(pos)
    
    # Using a simple proportional law to generate desired roll and pitch.
    # For small angles: roll_des ≈ (1/g)*(Kp_att * error_y), pitch_des ≈ -(1/g)*(Kp_att * error_x)
    roll_des = (1/g) * (Kp_att * pos_error[1])
    pitch_des = -(1/g) * (Kp_att * pos_error[0])
    yaw_des = 0.0  # desired yaw
    
    # --- Construct the Reference State ---
    # Reference state: [target position, zero velocity, desired attitude, zero angular velocity]
    ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                    0, 0, 0,
                    roll_des, pitch_des, yaw_des,
                    0, 0, 0])
    
    # --- Solve the MPC Problem ---
    # Pass pid_controller=None to disable separate PID altitude correction.
    try:
        u_opt, x_pred = mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=None)
    except Exception as e:
        print("MPC error:", e)
        u_opt = np.zeros(4)
    control_log.append(u_opt)
    
    # Extract control commands.
    delta_thrust = u_opt[0]
    roll_torque  = u_opt[1]
    pitch_torque = u_opt[2]
    yaw_torque   = u_opt[3]
    
    # Compute total thrust with gravitational compensation.
    total_thrust = delta_thrust + m_drone * g
    total_thrust = min(total_thrust, m_drone * g * 1.5)
    u_fin = [total_thrust, roll_torque, pitch_torque, yaw_torque]
    
    # --- Force/Torque Allocation ---
    f0 = total_thrust / 4 + roll_torque / (2 * l_arm) - pitch_torque / (2 * l_arm) - yaw_torque / 4
    f1 = total_thrust / 4 + roll_torque / (2 * l_arm) + pitch_torque / (2 * l_arm) + yaw_torque / 4
    f2 = total_thrust / 4 - roll_torque / (2 * l_arm) + pitch_torque / (2 * l_arm) - yaw_torque / 4
    f3 = total_thrust / 4 - roll_torque / (2 * l_arm) - pitch_torque / (2 * l_arm) + yaw_torque / 4
    force_log.append([f0, f1, f2, f3])
    
    # --- Force Transformation from Body to World Frame ---
    forces_body = [[0, 0, f0],
                   [0, 0, f1],
                   [0, 0, f2],
                   [0, 0, f3]]
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    bodyMatrix[3] = 0; bodyMatrix[7] = 0; bodyMatrix[11] = 0
    forces_world = [sim.multiplyVector(bodyMatrix, force) for force in forces_body]
    
    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandles[i], forces_world[i], [0, 0, 0])
    
    # Rotate the propellers for visual effect.
    velocities = [-100, 100, -100, 100]
    for i, v in enumerate(velocities):
        sim.setJointTargetVelocity(jointHandles[i], v)
    
    sim.step()
    
    # Safety check: terminate if position error becomes excessive.
    if np.linalg.norm(pos_error) >= 100:
        save_logs(time_log, state_log, control_log, force_log, target_log, filename="simulation_logs.csv")
        sim.stopSimulation()
        break

# =============================================================================
# SAVE LOGS & STOP SIMULATION
# =============================================================================

save_logs(time_log, state_log, control_log, force_log, target_log, filename="simulation_logs.csv")
sim.stopSimulation()
