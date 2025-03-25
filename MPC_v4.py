# Description: This script implements a simple MPC controller for a quadrotor in CoppeliaSim.
# The quadrotor is controlled to track the position of a target object in the scene.
# The MPC controller is used to generate control inputs for the quadrotor based on a simple linearized model.
# The MPC problem is solved using CVXPY, a convex optimization library.
# The script also uses a PID controller for altitude correction to improve tracking performance.
#



import noise
import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from PID_Controller import PIDController
import pandas as pd
from save_parameters import save_parameters
from save_logs import save_logs
from MPC_controller import mpc_controller
import time

# Connect to the remote API server.
client = RemoteAPIClient() 
sim = client.require('sim')

# Simulation time.
simualtion_time = 50    # Total simulation time [s]
kp_param = 0.8
ki_param = 0.1
kd_param = 1.15
# Data log initialization.
time_log = []      # List to log simulation time
state_log = []     # List to log state vectors (x0)
control_log = []   # List to log control inputs (u_opt)
force_log = []     # List to log forces applied to propellers
target_log = []    # Target position logs


# Get handles for the drone, target, and propellers.
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
propellerHandle = [None] * 4
jointHandle = [None] * 4
for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')


# Set the target object's initial position.
sim.setObjectPosition(targetHandle, -1, [0.0, 0.0, 2.0])
# Set the drone's initial position.
sim.setObjectPosition(droneHandle, -1, [0.0, 0.0, 1.0])


# Set simulation to stepping mode.
sim.setStepping(True)
sim.startSimulation()
#------------------------------------------------------------------
# Define model parameters.
dt = sim.getSimulationTimeStep()        # Time step [s]
N = 10           # MPC prediction horizon
m_drone = 0.52    # Drone mass [kg]
g = 9.81         # Gravitational acceleration [m/s²]
I_x = +6.667e-05           # moment of inertia about x (roll)
I_y = +0.00753           # moment of inertia about y (pitch)
I_z = +0.00753       # Moment of inertia about z (yaw)
#------------------------------------------------------------------
# Construct continuous-time A and B matrices.
A = np.zeros((12, 12))
# Position integration.
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
# Translational dynamics: linearized (small-angle) approximations.
A[3, 7] = g    # x acceleration from pitch.
A[4, 6] = -g   # y acceleration from roll.
# z acceleration does not depend on state.
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

# Discretize the system (Euler discretization):
A_d = np.eye(12) + A * dt
B_d = B * dt

#------------------------------------------------------------------
# Define cost matrices.
Q = np.diag([0.3, 0.3, 3.0,# x y z
             0.01, 0.01, 0.1,# vx vy vz
             0.25, 0.25, 0.1, # roll pitch yaw
             0.0, 0.0, 0.0,]) # p q r
R = np.diag([0.1, 0.4, 0.4, 0.1])


# Input bounds for [delta_thrust, roll torque, pitch torque, yaw torque].
u_min = np.array([ -m_drone*g, -0.015, -0.015, -1.0])
u_max = np.array([ 1.0,  0.015,  0.015,  1.0])

# Create an instance of the PID controller for altitude correction (optional).
pid_z = PIDController(kp=kp_param, ki=ki_param, kd=kd_param, dt=dt)

# Arm length for force allocation (distance from center to each propeller).
l_arm = 0.13
save_parameters(Q, R, N, pid_z.get_parameters(), filename="simulation_parameters.txt")
#------------------------------------------------------------------
# Simulation loop.
while (t := sim.getSimulationTime()) < simualtion_time:
    print(f"Simulation time: {t:.2f} s")
    
    # --- State Estimation ---
    # Get current position and linear velocity.
    pos = sim.getObjectPosition(droneHandle, -1)  # [x, y, z] in world frame.
    linVel, _ = sim.getObjectVelocity(droneHandle)  # [vx, vy, vz].
    # Get orientation (Euler angles: roll, pitch, yaw).
    ori = sim.getObjectOrientation(droneHandle, -1)
    # Get angular velocity.
    _, angVel = sim.getObjectVelocity(droneHandle)
    
    # Construct the state vector:
    x0 = np.array([pos[0], pos[1], pos[2],
                   linVel[0], linVel[1], linVel[2],
                   ori[0], ori[1], ori[2],
                   angVel[0], angVel[1], angVel[2]])
    """
    x0[0] = noise.add_measurement_noise(x0[0], 0.1)
    x0[1] = noise.add_measurement_noise(x0[1], 0.1)
    x0[2] = noise.add_measurement_noise(x0[2], 0.1)  # Simulate sensor noise.
    """
        # --- Log the current state and time ---
    time_log.append(t)
    state_log.append(x0)
    
    # --- Define the Reference State ---
    # The target state: desired position (from target object), zero velocities, level attitude and no angular rates.


    """ 
    targetObjPos = [
        4 * np.sin(0.4*t),  # Sine wave movement for x
        4 * np.cos(0.4*t),  # Sine wave movement for y
        3.0             # Fixed altitude (z)
    ]
    sim.setObjectPosition(targetHandle, -1, targetObjPos)  """
    
    


    #------------------------------------------------------------------



    targetPos = sim.getObjectPosition(targetHandle, -1)
    target_log.append(targetPos)
    

    
    ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0])
        


    
    # --- Solve the MPC Problem with PID Altitude Correction ---
    try:
        u_opt, x_pred = mpc_controller(x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=pid_z)
    except Exception as e:
        print("MPC error:", e)
        u_opt = np.zeros(4)
    
    # --- Extract Control Commands ---
    # u_opt = [delta_thrust, roll_torque, pitch_torque, yaw_torque]
    delta_thrust = u_opt[0]
    roll_torque  = u_opt[1]
    pitch_torque = u_opt[2]
    yaw_torque   = u_opt[3]
        # --- Log the control input ---
    
    
    
    # Compute total thrust command: add the gravitational compensation.
    total_thrust =  delta_thrust  + m_drone*g 
    total_thrust = min(total_thrust , m_drone*g*1.5) # Limit the maximum thrust. 


    #Log the control input
    u_fin = [total_thrust, roll_torque, pitch_torque, yaw_torque] # without gravitational compensation
    control_log.append(u_opt)
    #print(total_thrust)
    
    # --- Force/Torque Allocation ---
    # For a quadrotor in an "X" configuration, one common allocation is:
    f0 = total_thrust/4 + roll_torque/(2*l_arm) - pitch_torque/(2*l_arm) - yaw_torque/4
    f1 = total_thrust/4 + roll_torque/(2*l_arm) + pitch_torque/(2*l_arm) + yaw_torque/4
    f2 = total_thrust/4 - roll_torque/(2*l_arm) + pitch_torque/(2*l_arm) - yaw_torque/4
    f3 = total_thrust/4 - roll_torque/(2*l_arm) - pitch_torque/(2*l_arm) + yaw_torque/4

    # --- Log the forces applied to propellers ---
    force_log.append([f0, f1, f2, f3])
    
    
    # For simplicity, assume each propeller’s force acts along the drone’s local z-axis.
    forces_body = [
        [0, 0, f0],
        [0, 0, f1],
        [0, 0, f2],
        [0, 0, f3]
    ]
    
    # Transform the force vectors from the body frame to the world frame.
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    # Zero out the translation part (indices 3, 7, 11 in a 0-indexed list).
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0
    forces_world = []
    for force in forces_body:
        # Convert force to world frame.
        force_world = sim.multiplyVector(bodyMatrix, force)
        forces_world.append(force_world)
    
    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], forces_world[i], [0, 0, 0])
    
    # Step the simulation.
    sim.step()
# After simulation, save the logs.
timestr = time.strftime("%d%m%Y-%H%M%S")
#filename = f"simulation_logs_{timestr}.csv"
filename = f"simulation_logs.csv"
save_logs(time_log, state_log, control_log, force_log, target_log, filename=filename)




sim.stopSimulation()
