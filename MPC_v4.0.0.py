
# v4.0 Linear MPC control - PID on altitude Z

import functions.noise as noise
import numpy as np
import cvxpy as cp  # For MPC optimization
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functions.PID_Controller import PIDController
from functions.save_parameters import save_parameters
from functions.save_logs import save_logs
from functions.mpc_dist import mpc_controller
from functions.patterns import PatternGenerator
import time

# =============================================================================
# USER CONFIGURATION
# =============================================================================

LOG_FILENAME = "simulation_logs.csv"
SIMULATION_TIME = 50.0  # Total simulation time [s]
PATTERN_ID = 16  # pattern 0: Christmas tree, 1: rectangular
INITIAL_TARGET_POSITION = [0.0, 0.0, 1.0]
INITIAL_DRONE_POSITION = [0.0, 0.0, 1.0]

# Sensor noise simulation toggle: set to True to enable measurement noise simulation
SIMULATE_SENSOR_NOISE = False
# Standard deviation (std) for measurement noise applied to position measurements [m]
SENSOR_NOISE_STD = 0.1

PID_Z_KP = 0.8
PID_Z_KI = 0.1
PID_Z_KD = 1.15
DISTURBANCE_INTEGRATOR_GAIN = 0.3

MPC_PREDICTION_HORIZON = 10
DRONE_MASS = 3.61  # [kg]
GRAVITY = 9.81  # [m/s^2]
INERTIA_X = 0.4815
INERTIA_Y = 0.4815
INERTIA_Z = 0.5778
ARM_LENGTH = 0.13  # Distance from center to each propeller [m]
LINEAR_DRAG_COEFF = 0.0001

Q_WEIGHTS = [
    0.5, 0.5, 3.0,    # x, y, z
    0.05, 0.05, 0.1,  # vx, vy, vz
    0.75, 0.75, 0.1,  # roll, pitch, yaw
    0.0, 0.0, 0.0,    # p, q, r
]
R_WEIGHTS = [0.1, 0.05, 0.05, 0.1]

DELTA_THRUST_MAX = 1.0
ROLL_TORQUE_LIMIT = 1.15
PITCH_TORQUE_LIMIT = 1.15
YAW_TORQUE_LIMIT = 1.0

ATTITUDE_POSITION_GAIN = 0.1
MAX_THRUST_FACTOR = 1.5
MAX_DISTANCE_THRESHOLD = 100.0

PROPELLER_TARGET_VELOCITIES = (-100, 100, -100, 100)

APPLY_WIND = True
USE_WIND_X_COMPONENT = True
WIND_FORCE_X_RANGE = (1, 5)    # Low inclusive, high exclusive
USE_WIND_Y_COMPONENT = False
WIND_FORCE_Y_RANGE = (5, 16)



PARAMETERS_OUTPUT = "simulation_parameters.txt"

# =============================================================================
# INITIALIZATION & CONNECTION SETUP
# =============================================================================

# Connect to the remote API server.
client = RemoteAPIClient()
sim = client.require('sim')

# =============================================================================
# SIMULATION PARAMETERS & LOG INITIALIZATION
# =============================================================================

d_hat = np.zeros(3)


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
sim.setObjectPosition(targetHandle, -1, INITIAL_TARGET_POSITION)
# Set the drone's initial position.
sim.setObjectPosition(droneHandle, -1, INITIAL_DRONE_POSITION)

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
inertiaMatrix, _ = sim.getShapeInertia(droneHandle)
print(inertiaMatrix)
pattern_gen = PatternGenerator(PATTERN_ID, SIMULATION_TIME)

# Construct continuous-time A and B matrices.
A = np.zeros((12, 12))
# A matrix
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
A[3, 3] = -LINEAR_DRAG_COEFF / DRONE_MASS
A[3, 7] = GRAVITY
A[4, 4] = -LINEAR_DRAG_COEFF / DRONE_MASS
A[4, 6] = -GRAVITY
A[5, 5] = -LINEAR_DRAG_COEFF / DRONE_MASS
A[6, 9] = 1
A[7, 10] = 1
A[8, 11] = 1
A[9, 9] = -LINEAR_DRAG_COEFF / INERTIA_X
A[10, 10] = -LINEAR_DRAG_COEFF / INERTIA_Y
A[11, 11] = -LINEAR_DRAG_COEFF / INERTIA_Z
# Angular accelerations (rows 9-11) are driven by inputs only.
B = np.zeros((12, 4))
B[5, 0] = 1.0 / DRONE_MASS  # z acceleration from thrust deviation.
B[9, 1] = 1.0 / INERTIA_X        # roll acceleration from roll torque.
B[10, 2] = 1.0 / INERTIA_Y         # pitch acceleration from pitch torque.
B[11, 3] = 1.0 / INERTIA_Z          # yaw acceleration from yaw torque.

# Discretize the system (Euler discretization).
A_d = np.eye(12) + A * dt
B_d = B * dt

# =============================================================================
# COST MATRICES & CONTROL LIMITS
# =============================================================================

# Define cost matrices.
Q = np.diag(Q_WEIGHTS)
R = np.diag(R_WEIGHTS)

# Input bounds for [delta_thrust, roll torque, pitch torque, yaw torque].
u_min = np.array([
    -DRONE_MASS * GRAVITY,
    -ROLL_TORQUE_LIMIT,
    -PITCH_TORQUE_LIMIT,
    -YAW_TORQUE_LIMIT,
])
u_max = np.array([
    DELTA_THRUST_MAX,
    ROLL_TORQUE_LIMIT,
    PITCH_TORQUE_LIMIT,
    YAW_TORQUE_LIMIT,
])

# Create an instance of the PID controller for altitude correction.
pid_z = PIDController(kp=PID_Z_KP, ki=PID_Z_KI, kd=PID_Z_KD, dt=dt)

# Save simulation parameters.
save_parameters(Q, R, MPC_PREDICTION_HORIZON, pid_z.get_parameters(), filename=PARAMETERS_OUTPUT)

# =============================================================================
# SIMULATION LOOP
# =============================================================================
while (t := sim.getSimulationTime()) < SIMULATION_TIME:
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
    # Add measurement noise if enabled.
    if SIMULATE_SENSOR_NOISE:
        # `add_measurement_noise` expects an array-like input; convert scalar position
        # entries into 1-element arrays, apply noise, then extract scalar back.
        x0[0] = float(noise.add_measurement_noise(np.array([x0[0]]), SENSOR_NOISE_STD)[0])
        x0[1] = float(noise.add_measurement_noise(np.array([x0[1]]), SENSOR_NOISE_STD)[0])
        x0[2] = float(noise.add_measurement_noise(np.array([x0[2]]), SENSOR_NOISE_STD)[0])
    
    # Log the current state and time.
    time_log.append(t)
    state_log.append(x0)
    
    
    
    targetObjPos = pattern_gen.get_position(t)
    sim.setObjectPosition(targetHandle, -1, targetObjPos)

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
    # For small angles: roll_des ~ (1/g)*(ATTITUDE_POSITION_GAIN * error_y), pitch_des ~ -(1/g)*(ATTITUDE_POSITION_GAIN * error_x)
    roll_des = (1 / GRAVITY) * (ATTITUDE_POSITION_GAIN * pos_error[1])
    pitch_des = -(1 / GRAVITY) * (ATTITUDE_POSITION_GAIN * pos_error[0])
    yaw_des = 0.0  # desired yaw
    ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                    0, 0, 0,
                    roll_des, pitch_des, yaw_des,
                    0, 0, 0])
    
    distance_log.append(distance)

    pos_err = np.array(pos[:3]) - np.array(targetPos[:3])
    d_hat += DISTURBANCE_INTEGRATOR_GAIN * pos_err * dt


    # --- Solve the MPC Problem with PID Altitude Correction ---
    try:
        u_opt, x_pred = mpc_controller(
            x0, A_d, B_d, ref, MPC_PREDICTION_HORIZON, Q, R, u_min, u_max,
            pid_controller_z=pid_z, d_hat=d_hat
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
    total_thrust = delta_thrust + DRONE_MASS * GRAVITY
    total_thrust = min(total_thrust, DRONE_MASS * GRAVITY * MAX_THRUST_FACTOR)  # Limit the maximum thrust.
    # --- Force/Torque Allocation ---
    f0 = total_thrust / 4 + roll_torque / (2 * ARM_LENGTH) - pitch_torque / (2 * ARM_LENGTH) - yaw_torque / 4
    f1 = total_thrust / 4 + roll_torque / (2 * ARM_LENGTH) + pitch_torque / (2 * ARM_LENGTH) + yaw_torque / 4
    f2 = total_thrust / 4 - roll_torque / (2 * ARM_LENGTH) + pitch_torque / (2 * ARM_LENGTH) - yaw_torque / 4
    f3 = total_thrust / 4 - roll_torque / (2 * ARM_LENGTH) - pitch_torque / (2 * ARM_LENGTH) + yaw_torque / 4
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

    # Add wind forces when enabled.
    if APPLY_WIND:
        wind_x = np.random.randint(*WIND_FORCE_X_RANGE) if USE_WIND_X_COMPONENT else 0
        wind_y = np.random.randint(*WIND_FORCE_Y_RANGE) if USE_WIND_Y_COMPONENT else 0
        wind_vector = [wind_x, wind_y, 0]
        sim.addForceAndTorque(droneHandle, wind_vector, [0, 0, 0])

    # --- Apply Forces to Propellers ---
    for i in range(4):
        sim.addForceAndTorque(propellerHandle[i], forces_world[i], [0, 0, 0])
    
    # Rotate the propellers (visual effect)
    for joint, velocity in zip(jointHandle, PROPELLER_TARGET_VELOCITIES):
        sim.setJointTargetVelocity(joint, velocity)

    # Step the simulation.
    sim.step()
    if distance >= MAX_DISTANCE_THRESHOLD:
        save_logs(time_log, state_log, control_log, force_log, target_log, distance_log, filename=LOG_FILENAME)
        sim.stopSimulation()
        break

# =============================================================================
# SAVE LOGS & STOP SIMULATION
# =============================================================================


save_logs(time_log, state_log, control_log, force_log, target_log, distance_log, filename=LOG_FILENAME)

sim.stopSimulation()
