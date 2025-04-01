#----- v4.5: Trajectory generatation-- see trajectory.py
#----- v4.5.1 PID control on x y


import functions.noise as noise
import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functions.PID_Controller import PIDController
import pandas as pd
from functions.save_parameters import save_parameters
from functions.save_logs import save_logs
from functions.MPC_controller import mpc_controller
import time
import functions.trajectory as trajectory

# =============================================================================
# INITIALIZATION & CONNECTION SETUP
# =============================================================================

# Connect to the remote API server.
client = RemoteAPIClient()
sim = client.require('sim')

timestr = time.strftime("%d%m%Y-%H%M%S")
log_filename = f"simulation_logs.csv"

# =============================================================================
# SIMULATION PARAMETERS & LOG INITIALIZATION
# =============================================================================

simualtion_time = 50  # Total simulation time [s]

init_target_pos = [-2.5, -2.5, 1.0] # Initial target position x y z
init_drone_pos =  [-2.5, -2.5, 1.0] # Initial drone position x y z


# PID parameters.
kp_z = 0.8
ki_z = 0.1 
kd_z = 1.15


kp_x = 1.0  
ki_x = 0.02  
kd_x = 0.5


kp_y = 1.0
ki_y = 0.02        
kd_y = 0.5

# Data log initialization.
time_log = []      # List to log simulation time
state_log = []     # List to log state vectors (x0)
control_log = []   # List to log control inputs (u_opt)
force_log = []     # List to log forces applied to propellers
target_log = []    # Target position logs
distance_log = []




# =============================================================================
# GET HANDLES FOR OBJECTS
# =============================================================================

# Get handles for the drone and target.
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')

# Get handles for propellers and their joints.
propellerHandle = [None] * 4
jointHandle = [None] * 4
for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')

# =============================================================================
# SET INITIAL POSITIONS
# =============================================================================

# Set the target object's initial position.
sim.setObjectPosition(targetHandle, -1, init_target_pos)
# Set the drone's initial position.
sim.setObjectPosition(droneHandle, -1,  init_drone_pos)

# =============================================================================
# START SIMULATION
# =============================================================================

# Set simulation to stepping mode.
sim.setStepping(True)
sim.startSimulation()

# =============================================================================
# SYSTEM MODEL PARAMETERS & DISCRETIZATION
# =============================================================================

# Define model parameters.
dt = sim.getSimulationTimeStep()  # Time step [s]
N = 10                            # MPC prediction horizon
m_drone = 0.52                # Drone mass [kg]
g = 9.81                          # Gravitational acceleration [m/s²]
I_x = +6.667e-05                  # moment of inertia about x (roll)
I_y = +0.00753                    # moment of inertia about y (pitch)
I_z = +0.00753                    # moment of inertia about z (yaw)
l_arm = 0.13 # Arm length for force allocation (distance from center to each propeller). [m]

# Construct continuous-time A and B matrices.
A = np.zeros((12, 12))
# Position integration.
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
# Translational dynamics: linearized (small-angle) approximations.
A[3, 7] = g    # x acceleration from pitch.
A[4, 6] = -g   # y acceleration from roll.
# Attitude kinematics.
A[6, 9] = 1    # roll dot = p.
A[7,10] = 1    # pitch dot = q.
A[8,11] = 1    # yaw dot = r.

# Angular accelerations (rows 9-11) are driven by inputs only.
B = np.zeros((12, 4))
B[5, 0] = 1.0 / m_drone  # z acceleration from thrust deviation.
B[9, 1] = 1.0            # roll acceleration from roll torque.
B[10,2] = 1.0            # pitch acceleration from pitch torque.
B[11,3] = 1.0            # yaw acceleration from yaw torque.

# Discretize the system (Euler discretization).
A_d = np.eye(12) + A * dt
B_d = B * dt

# =============================================================================
# COST MATRICES & CONTROL LIMITS
# =============================================================================

# Define cost matrices.
Q = np.diag([0.3, 0.3, 3.0,    # x, y, z
             0.01, 0.01, 0.1,  # vx, vy, vz
             0.25, 0.25, 0.1,  # roll, pitch, yaw
             0.0, 0.0, 0.0])   # p, q, r

R = np.diag([0.1, 0.4, 0.4, 0.1])

# Input bounds for [delta_thrust, roll torque, pitch torque, yaw torque].
u_min = np.array([-m_drone * g, -0.015, -0.015, -1.0])
u_max = np.array([1.0, 0.015, 0.015, 1.0])

# Create an instance of the PID controller for altitude correction.
pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)


# Save simulation parameters.
save_parameters(Q, R, N, pid_z.get_parameters(), filename="simulation_parameters.txt")

pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)
pid_x = PIDController(kp=kp_x, ki=ki_x, kd=kd_x, dt=dt)
pid_y = PIDController(kp=kp_y, ki=ki_y, kd=kd_y, dt=dt)

# =============================================================================
# TRAJECTORY GENERATION
# =============================================================================
x_min = -2,5
x_max = 2.5
y_min = -2.5
y_max = 2.5



_,positions,velocities,accelerations = trajectory.generate_landmower_trajectory(-2.5,2.5,-2.5,2.5,0.5,10,0.5,dt,1.0)


positions = positions.tolist()
velocities = velocities.tolist()
accelerations.tolist()
step_count = 0

# =============================================================================
# SIMULATION LOOP
# =============================================================================

while (t := sim.getSimulationTime()) < simualtion_time:
    print(f"Simulation time: {t:.2f} s")
    
    # --- State Estimation ---
    pos = sim.getObjectPosition(droneHandle, -1)  # [x, y, z] in world frame.
    linVel, _ = sim.getObjectVelocity(droneHandle)  # [vx, vy, vz]
    ori = sim.getObjectOrientation(droneHandle, -1)  # [roll, pitch, yaw]
    _, angVel = sim.getObjectVelocity(droneHandle)   # Angular velocities.
    
    # Construct the state vector.
    x0 = np.array([
        pos[0], pos[1], pos[2],
        linVel[0], linVel[1], linVel[2],
        ori[0], ori[1], ori[2],
        angVel[0], angVel[1], angVel[2]
    ])

    
    # Log the current state and time.
    time_log.append(t)
    state_log.append(x0)

    # --- Define the Reference State ---
    sim.setObjectPosition(targetHandle, -1, positions[step_count][:])
    targetPos = sim.getObjectPosition(targetHandle, -1)
    error_x = targetPos[0] - pos[0]
    error_x = max(-1.0, min(1.0, error_x))
    error_y = targetPos[1] - pos[1]
    error_y = max(-1.0, min(1.0, error_y))
    ax_des = pid_x.update(error_x)
    ay_des = pid_y.update(error_y)
    targetVel = velocities[step_count][:]
    targetAcc = accelerations[step_count][:]
    target_log.append(targetPos)
    ref = np.array([
        targetPos[0], targetPos[1], targetPos[2],
        targetVel[0]+ax_des, targetVel[1]+ay_des, targetVel[2],
        0, 0, 0,
        0, 0, 0
    ])

    # -- Compute the distance to the target --
    x_err = targetPos[0] - pos[0]
    y_err = targetPos[1] - pos[1]
    z_err = targetPos[2] - pos[2]
    distance = np.sqrt(x_err**2 + y_err**2 + z_err**2)
    distance_log.append(distance)

    # --- Solve the MPC Problem with PID Altitude Correction ---
    try:
        u_opt, x_pred = mpc_controller(
            x0, A_d, B_d, ref, N, Q, R, u_min, u_max, 
            pid_controller=pid_z,
        )
    except Exception as e:
        print("MPC error:", e)
        u_opt = np.zeros(4)

    # Extract control commands.
    delta_thrust = u_opt[0]
    roll_torque  = u_opt[1]
    pitch_torque = u_opt[2]
    yaw_torque   = u_opt[3]
    control_log.append(u_opt)

    # Compute total thrust command: add gravitational compensation.
    total_thrust = delta_thrust + m_drone * g
    total_thrust = min(total_thrust, m_drone * g * 1.5)  # Limit the maximum thrust.
    u_fin = [total_thrust, roll_torque, pitch_torque, yaw_torque]

    # --- Force/Torque Allocation ---
    f0 = total_thrust / 4 + roll_torque / (2 * l_arm) - pitch_torque / (2 * l_arm) - yaw_torque / 4
    f1 = total_thrust / 4 + roll_torque / (2 * l_arm) + pitch_torque / (2 * l_arm) + yaw_torque / 4
    f2 = total_thrust / 4 - roll_torque / (2 * l_arm) + pitch_torque / (2 * l_arm) - yaw_torque / 4
    f3 = total_thrust / 4 - roll_torque / (2 * l_arm) - pitch_torque / (2 * l_arm) + yaw_torque / 4
    force_log.append([f0, f1, f2, f3])

    # --- Force Transformation from Body to World Frame ---
    forces_body = [
        [0, 0, f0],
        [0, 0, f1],
        [0, 0, f2],
        [0, 0, f3]
    ]
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    # Zero out the translation part.
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0

    forces_world = []
    for force in forces_body:
        force_world = sim.multiplyVector(bodyMatrix, force)
        forces_world.append(force_world)

    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], forces_world[i], [0, 0, 0])
    
    #Rotate the propellers (visual effect)
    sim.setJointTargetVelocity(jointHandle[0], -100)
    sim.setJointTargetVelocity(jointHandle[1], 100)
    sim.setJointTargetVelocity(jointHandle[2], -100)
    sim.setJointTargetVelocity(jointHandle[3], 100)

    # Step the simulation.
    sim.step()
    step_count +=1

# =============================================================================
# SAVE LOGS & STOP SIMULATION
# =============================================================================


save_logs(time_log, state_log, control_log, force_log, target_log,distance_log, filename=log_filename)

sim.stopSimulation()
