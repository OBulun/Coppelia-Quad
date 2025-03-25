from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
client = RemoteAPIClient() 
sim = client.require('sim')
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

# === Global PID Gains  ===
# Position (outer loop)
kp_z, ki_z, kd_z = 1.0, 0.0, 0.1
kp_x, ki_x, kd_x = 0.5, 0.0, 0.05  
kp_y, ki_y, kd_y = 0.5, 0.0, 0.05
# Attitude (inner loop)
kp_roll, ki_roll, kd_roll = 0.8, 0.0, 0.05
kp_pitch, ki_pitch, kd_pitch = 0.8, 0.0, 0.05
kp_yaw, ki_yaw, kd_yaw = 0.6, 0.0, 0.04
# === Global State Variables for PID (position and attitude) ===
# Position errors
last_error_x = last_error_y = last_error_z = 0
cum_error_x = cum_error_y = cum_error_z = 0

# Attitude errors
last_error_roll = last_error_pitch = last_error_yaw = 0
cum_error_roll = cum_error_pitch = cum_error_yaw = 0

res = np.array([None, None, None, None])
force = np.array([None, None, None, None])
torque = np.array([None, None, None, None])

sim.setObjectPosition(droneHandle, [0.0, 0.0, 0.2])
sim.setObjectPosition(targetHandle, [0.0, 0.0, 2])

sim.setStepping(True)
sim.startSimulation()
while (t := sim.getSimulationTime()) < 500:
    print(f'Simulation time: {t:.2f} [s]')


    # --- Time Step ---
    ts = sim.getSimulationTimeStep()

    # --- Get Current Positions and Orientation ---
    # Get positions in world frame:
    targetPos = sim.getObjectPosition(targetHandle, -1)  # [x, y, z]
    dronePos  = sim.getObjectPosition(droneHandle, -1)
    # Get current orientation (Euler angles: [roll, pitch, yaw]) in world frame:
    droneOri  = sim.getObjectOrientation(droneHandle, -1)

    # --- Outer Loop: Position Control (World Frame) ---
    # Compute position errors:
    error_x = targetPos[0] - dronePos[0]
    error_y = targetPos[1] - dronePos[1]
    error_z = targetPos[2] - dronePos[2]

    # --- Altitude (z) PID ---
    cum_error_z += error_z * ts
    d_error_z = (error_z - last_error_z) / ts
    last_error_z = error_z
    # u_z is the desired vertical thrust (force) component:
    u_z = kp_z * error_z + ki_z * cum_error_z + kd_z * d_error_z

    # --- Horizontal Position to Attitude Conversion ---
    # A simple approach: use proportional control to generate desired tilt angles.
    # For many quadrotors: tilting forward (pitch positive) moves in x, tilting sideways (roll positive) moves in y.
    # Here, we directly map the position errors to desired pitch and roll angles.
    desired_pitch = kp_x * error_x  # (adjust gain as needed)
    desired_roll  = kp_y * error_y

    # --- Inner Loop: Attitude (Orientation) Control ---
    # Current attitudes:
    current_roll  = droneOri[0]
    current_pitch = droneOri[1]
    current_yaw   = droneOri[2]

    # For yaw, assume we want a fixed heading (for example, desired_yaw = 0)
    desired_yaw = 0

    # Compute attitude errors:
    error_roll  = desired_roll  - current_roll
    error_pitch = desired_pitch - current_pitch
    error_yaw   = desired_yaw   - current_yaw

    # Roll PID:
    cum_error_roll += error_roll * ts
    d_error_roll = (error_roll - last_error_roll) / ts
    last_error_roll = error_roll
    u_roll = kp_roll * error_roll + ki_roll * cum_error_roll + kd_roll * d_error_roll

    # Pitch PID:
    cum_error_pitch += error_pitch * ts
    d_error_pitch = (error_pitch - last_error_pitch) / ts
    last_error_pitch = error_pitch
    u_pitch = kp_pitch * error_pitch + ki_pitch * cum_error_pitch + kd_pitch * d_error_pitch

    # Yaw PID:
    cum_error_yaw += error_yaw * ts
    d_error_yaw = (error_yaw - last_error_yaw) / ts
    last_error_yaw = error_yaw
    u_yaw = kp_yaw * error_yaw + ki_yaw * cum_error_yaw + kd_yaw * d_error_yaw

    # --- Force and Torque Allocation ---
    # For a quadrotor, the total vertical thrust is u_z.
    # Differential forces produce pitch, roll, and yaw torques.
    # One common allocation (for an "X" configuration) is:
    #
    #   Propeller 0: u_z/4 - u_pitch + u_roll - u_yaw
    #   Propeller 1: u_z/4 - u_pitch - u_roll + u_yaw
    #   Propeller 2: u_z/4 + u_pitch - u_roll - u_yaw
    #   Propeller 3: u_z/4 + u_pitch + u_roll + u_yaw
    #
    # These numbers may need to be adjusted for your drone’s geometry and dynamics.
    f0 = u_z - u_pitch + u_roll - u_yaw
    f1 = u_z - u_pitch - u_roll + u_yaw
    f2 = u_z + u_pitch - u_roll - u_yaw
    f3 = u_z + u_pitch + u_roll + u_yaw

    # To convert forces defined in the drone's body frame into world frame,
    # we use the drone’s transformation matrix. First, zero out the translation.
    bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
    bodyMatrix[3] = 0  # zero out x-translation
    bodyMatrix[7] = 0  # zero out y-translation
    bodyMatrix[11] = 0 # zero out z-translation

    # Assume each propeller applies its force along the drone’s local z axis.
    forces = []
    for f in [f0, f1, f2, f3]:
        local_force = [0, 0, f]  # force vector in body frame
        world_force = sim.multiplyVector(bodyMatrix, local_force)
        forces.append(world_force)

    # For simplicity, we set torque to zero on each propeller
    # (if you wish to apply additional torque commands, you can compute and add them similarly).
    torque_world = [0, 0, 0]

    # Apply forces and torques to each propeller:
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], forces[i], torque_world)

    # --- Debug Output ---
    print(f'Pos Errors: x: {error_x:.2f}, y: {error_y:.2f}, z: {error_z:.2f}')
    print(f'Attitude Errors: roll: {error_roll:.2f}, pitch: {error_pitch:.2f}, yaw: {error_yaw:.2f}')
    print(f'Control Signals: u_z: {u_z:.2f}, u_pitch: {u_pitch:.2f}, u_roll: {u_roll:.2f}, u_yaw: {u_yaw:.2f}')

    sim.step()
sim.stopSimulation()