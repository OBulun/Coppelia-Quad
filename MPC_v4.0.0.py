
# v4.0 Linear MPC control - PID on altitude Z

import functions.noise as noise
import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functions.PID_Controller import PIDController
import pandas as pd
from functions.save_parameters import save_parameters
from functions.save_logs import save_logs
from functions.mpc_osqp import mpc_controller
import time

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

simualtion_time =30  # Total simulation time [s]

pattern = 0  # pattern 0:Christmas tree 1: rectangular

# PID parameters.
kp_z = 0.8
ki_z = 0.1
kd_z = 1.15

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
sim.setObjectPosition(targetHandle, -1, [0.0, 4.0, 1.0])
# Set the drone's initial position.
sim.setObjectPosition(droneHandle, -1, [0.0, 4.0, 1.0])

# =============================================================================
# START SIMULATION
# =============================================================================

# Set simulation to stepping mode.
sim.setStepping(True)
sim.startSimulation()

# =============================================================================
# SYSTEM MODEL PARAMETERS & DISCRETIZATION
# =============================================================================
"""
# Define model parameters.
dt = sim.getSimulationTimeStep()  # Time step [s]
N = 10                            # MPC prediction horizon
m_drone = 3.61                # Drone mass [kg]
g = 9.81                          # Gravitational acceleration [m/s²]
I_x = +0.4815                  # moment of inertia about x (roll)
I_y = +0.4815                 # moment of inertia about y (pitch)
I_z = +0.5778                     # moment of inertia about z (yaw)
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
B[9, 1] = 1.0 / I_x           # roll acceleration from roll torque.
B[10,2] = 1.0 / I_y           # pitch acceleration from pitch torque.
B[11,3] = 1.0 / I_z          # yaw acceleration from yaw torque.
"""

# Define model parameters.
dt = sim.getSimulationTimeStep()  # Time step [s]
N = 10                            # MPC prediction horizon
m_drone = 3.61                # Drone mass [kg]
g = 9.81                          # Gravitational acceleration [m/s²]
I_x = +0.4815                  # moment of inertia about x (roll)
I_y = +0.4815                 # moment of inertia about y (pitch)
I_z = +0.5778                    # moment of inertia about z (yaw)
l_arm = 0.13 # Arm length for force allocation (distance from center to each propeller). [m]
bf = 0.0001
inertiaMatrix, _ = sim.getShapeInertia(droneHandle)
print(inertiaMatrix)

# Construct continuous-time A and B matrices.
A = np.zeros((12, 12))
# A matrix
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
A[9, 9] = -bf / I_x
A[10, 10] = -bf / I_y
A[11, 11] = -bf / I_z
# Angular accelerations (rows 9-11) are driven by inputs only.
B = np.zeros((12, 4))
B[5, 0] = 1.0 / m_drone  # z acceleration from thrust deviation.
B[9, 1] = 1.0 / I_x        # roll acceleration from roll torque.
B[10,2] = 1.0  /I_y         # pitch acceleration from pitch torque.
B[11,3] = 1.0  /I_z          # yaw acceleration from yaw torque.

# Discretize the system (Euler discretization).
A_d = np.eye(12) + A * dt
B_d = B * dt

# =============================================================================
# COST MATRICES & CONTROL LIMITS
# =============================================================================

# Define cost matrices.
Q = np.diag([0.5, 0.5, 3.0,    # x, y, z
             0.05, 0.05, 0.1,  # vx, vy, vz
             0.75, 0.75, 0.1,  # roll, pitch, yaw
             0.0, 0.0, 0.0])   # p, q, r

R = np.diag([0.1, 0.05, 0.05, 0.1])

# Input bounds for [delta_thrust, roll torque, pitch torque, yaw torque].
u_min = np.array([-m_drone * g, -1.15, -1.15, -1.0])
u_max = np.array([1.0, 1.15, 1.15, 1.0])

# Create an instance of the PID controller for altitude correction.
pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)


# Save simulation parameters.
save_parameters(Q, R, N, pid_z.get_parameters(), filename="simulation_parameters.txt")

# =============================================================================
# SIMULATION LOOP
# =============================================================================
Kp_att = 0.1
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
    """ Simulate Sensor Noise """
    """ x0[0] = noise.add_measurement_noise(x0[0], 0.1)
    x0[1] = noise.add_measurement_noise(x0[1], 0.1)
    x0[2] = noise.add_measurement_noise(x0[2], 0.1) """  
    
    # Log the current state and time.
    time_log.append(t)
    state_log.append(x0)
    
    
    """# --- Target Position Update - Christmas Tree Pattern ---"""  
    if pattern == 0:
        end_time = simualtion_time-10
        if simualtion_time-t < 10:
            targetObjPos = [0, 0, 1.0+0.1*end_time]
        else:
        
            targetObjPos = [
                4*np.exp(-0.05*t) * np.sin(0.9*t),  # Sine wave movement for x
                4*np.exp(-0.05*t) * np.cos(0.9*t),  # Sine wave movement for y
                1.0+0.1*t             # Fixed altitude (z)
            ]
        sim.setObjectPosition(targetHandle, -1, targetObjPos)

    elif pattern == 1:
        """ # --- Target Position Update - Step  ---"""

        if t >=0.1 and t < 5:
            sim.setObjectPosition(targetHandle, -1, [2.0, 2.0, 2.0])
        elif t>= 5 and t<10:
            sim.setObjectPosition(targetHandle, -1, [-2.0, 2.0, 2.0])
        elif t>= 10 and t < 15:
            sim.setObjectPosition(targetHandle, -1, [-2.0, -2.0, 2.0])
        elif t>=15 and t< 20:
            sim.setObjectPosition(targetHandle, -1, [2.0, -2.0, 2.0])
        else:
            sim.setObjectPosition(targetHandle, -1, [0.0, 0.0, 2.0])
    elif pattern == 2:
        pass    
        
    

    # --- Define the Reference State ---
    targetPos = sim.getObjectPosition(targetHandle, -1)
    target_log.append(targetPos)

    
    # -- Compute the distance to the target --
    x_err = targetPos[0] - pos[0]
    y_err = targetPos[1] - pos[1]
    z_err = targetPos[2] - pos[2]
    distance = np.sqrt(x_err**2 + y_err**2 + z_err**2)
    pos_error = np.array(targetPos) - np.array(pos)
    # Using a simple proportional law to generate desired roll and pitch.
    # For small angles: roll_des ≈ (1/g)*(Kp_att * error_y), pitch_des ≈ -(1/g)*(Kp_att * error_x)
    roll_des = (1/g) * (Kp_att * pos_error[1])
    pitch_des = -(1/g) * (Kp_att * pos_error[0])
    yaw_des = 0.0  # desired yaw
    ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                    0, 0, 0,
                    roll_des, pitch_des, yaw_des,
                    0, 0, 0])
    
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

    #Add wind forces
    wind_x = random_int = np.random.randint(5, 16)
    wind_y = random_int = np.random.randint(5, 16)
    windVector = [0,0,0]#[wind_x,wind_y,0]
    sim.addForceAndTorque(droneHandle, windVector, [0, 0, 0])

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
    if distance >= 100:
        save_logs(time_log, state_log, control_log, force_log, target_log,distance_log, filename=log_filename)
        sim.stopSimulation()
        break

# =============================================================================
# SAVE LOGS & STOP SIMULATION
# =============================================================================


save_logs(time_log, state_log, control_log, force_log, target_log,distance_log, filename=log_filename)

sim.stopSimulation()
