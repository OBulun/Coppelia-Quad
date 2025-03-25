import numpy as np
from control import lqr  # Requires the python-control library
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random

# Connect to the server
client = RemoteAPIClient() 
sim = client.require('sim')


g = 9.81
m = 0.12  # mass of the drone [kg]
# For a simplified linear model, we define a 12-state system:
# x = [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]ᵀ
A = np.zeros((12, 12))
# Kinematics:
A[0, 3] = 1.0
A[1, 4] = 1.0
A[2, 5] = 1.0
# Translational dynamics (using small-angle approximations):
A[3, 7] = g   # x acceleration influenced by pitch
A[4, 6] = -g  # y acceleration influenced by roll
# Attitude dynamics (integration of angular rates):
A[6, 9]  = 1.0
A[7, 10] = 1.0
A[8, 11] = 1.0
# (z dynamics: z'' = (1/m)*F_total – g, is handled in the input)

B = np.zeros((12, 4))
# Control inputs u = [F_total, tau_phi, tau_theta, tau_psi]ᵀ.
B[5, 0]  = 1/m    # F_total affects vertical acceleration
B[9, 1]  = 1.0    # tau_phi affects roll dynamics
B[10, 2] = 1.0    # tau_theta affects pitch dynamics
B[11, 3] = 1.0    # tau_psi affects yaw dynamics

# ----------- Cost Matrices for LQR -----------
# Here we penalize state errors and control effort.
Q = np.eye(12)
R = np.eye(4)

# Compute the LQR gain matrix: u = -K*(x - x_ref)
K, S, E = lqr(A, B, Q, R)
print("Computed LQR Gain Matrix K:")
print(K)


#Handles
propellerHandle = np.array([None, None, None, None])   
jointHandle = np.array([None, None, None, None])
forceSensor = np.array([None, None, None, None])
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')
    forceSensor[i] = sim.getObject(f'/Quadcopter/propeller[{i}]')


sim.setStepping(True)
sim.startSimulation()
while (t := sim.getSimulationTime()) < 500:
    print(f'Simulation time: {t:.2f} [s]')


# --- Obtain State Measurements ---
    # Get drone position and orientation in the world frame:
    dronePos = sim.getObjectPosition(droneHandle, -1)      # [x, y, z]
    targetPos  = sim.getObjectPosition(targetHandle, -1)     # desired position
    
    # Get drone linear and angular velocities:
    # sim.getObjectVelocity returns a tuple: ([vx,vy,vz], [p,q,r])
    linVel, angVel = sim.getObjectVelocity(droneHandle)
    
    # Get current orientation (Euler angles: roll, pitch, yaw)
    droneOri = sim.getObjectOrientation(droneHandle, -1)
    
    # Construct the current state vector:
    # For positions (x, y, z), velocities (vx, vy, vz), attitudes (roll, pitch, yaw) and angular rates (p, q, r)
    x_state = np.array([
        dronePos[0], dronePos[1], dronePos[2],
        linVel[0],  linVel[1],  linVel[2],
        droneOri[0], droneOri[1], droneOri[2],
        angVel[0],  angVel[1],  angVel[2]
    ])
    
    # Define the desired state.
    # We want the drone to reach the target position with zero velocities and level attitude.
    x_ref = np.array([
        targetPos[0], targetPos[1], targetPos[2],
        0, 0, 0,         # zero linear velocities
        0, 0, 0,         # level orientation: roll, pitch, yaw = 0
        0, 0, 0          # zero angular rates
    ])
    
    error = x_state - x_ref
    
    # --- Compute Optimal Control Input ---
    # u = [F_total, tau_phi, tau_theta, tau_psi]
    u = -K.dot(error)
    F_total, tau_phi, tau_theta, tau_psi = u
    print("Optimal control input u:", u)
    
    # --- Force/Torque Allocation to Propellers ---
    # For a quadrotor in "X" configuration, one typical allocation is:
    # Propeller 0: F_total/4 - tau_phi/(2*l) - tau_theta/(2*l) - tau_psi/4
    # Propeller 1: F_total/4 - tau_phi/(2*l) + tau_theta/(2*l) + tau_psi/4
    # Propeller 2: F_total/4 + tau_phi/(2*l) - tau_theta/(2*l) + tau_psi/4
    # Propeller 3: F_total/4 + tau_phi/(2*l) + tau_theta/(2*l) - tau_psi/4
    # where l is the distance from the drone center to a propeller.
    l_arm = 0.13  # example arm length [m]
    f0 = F_total/4 - tau_phi/(2*l_arm) - tau_theta/(2*l_arm) - tau_psi/4
    f1 = F_total/4 - tau_phi/(2*l_arm) + tau_theta/(2*l_arm) + tau_psi/4
    f2 = F_total/4 + tau_phi/(2*l_arm) - tau_theta/(2*l_arm) + tau_psi/4
    f3 = F_total/4 + tau_phi/(2*l_arm) + tau_theta/(2*l_arm) - tau_psi/4
    
    # --- Transforming Forces to World Frame ---
    # Assume each propeller applies its force along the drone’s local z-axis.
    # Get the drone’s transformation matrix (body to world):
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    # Zero the translation part (indices 3, 7, 11 in a 0-indexed 12-element list)
    bodyMatrix[3] = 0
    bodyMatrix[7] = 0
    bodyMatrix[11] = 0
    
    forces = []
    for f in [f0, f1, f2, f3]:
        local_force = [0, 0, f]
        world_force = sim.multiplyVector(bodyMatrix, local_force)
        forces.append(world_force)
    
    # For this example we are not adding extra torques (they are implicitly applied via differential forces)
    zeroTorque = [0, 0, 0]
    
    # --- Apply Forces to Each Propeller ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], forces[i], zeroTorque)



    sim.step()
sim.stopSimulation()