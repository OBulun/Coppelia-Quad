import numpy as np
import cvxpy as cp  # For MPC optimization
import time
from scipy.optimize import minimize
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functions.PID_Controller import PIDController
from functions.save_parameters import save_parameters
from functions.save_logs import save_logs
from functions.MPC_controller import mpc_controller

def run_simulation(Q):
    """
    Run the MPC simulation using the given Q matrix (diagonal cost matrix)
    and return a cost computed from the integrated squared distance error 
    between the drone and the target.
    
    (Replace parts of this simulation code with your actual simulation as needed.)
    """
    # Connect to the remote API server.
    client = RemoteAPIClient()
    sim = client.require('sim')
    
    simualtion_time = 5  # Total simulation time [s]
    kp_z, ki_z, kd_z = 0.8, 0.1, 1.15

    # Initialize handles for drone, target, propellers and joints.
    droneHandle = sim.getObject('/Quadcopter')
    targetHandle = sim.getObject('/target')
    propellerHandle = [None] * 4
    jointHandle = [None] * 4
    for i in range(4):
        propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
        jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')

    # Reset initial positions.
    sim.setObjectPosition(targetHandle, -1, [0.0, 0.0, 1.0])
    sim.setObjectPosition(droneHandle, -1, [0.0, 0.0, 1.0])
    
    # Initialize logs.
    distance_log = []
    time_log = []
    state_log = []
    control_log = []
    force_log = []
    target_log = []
    
    # Ensure positions are reset.
    sim.setObjectPosition(targetHandle, -1, [0.0, 0.0, 1.0])
    sim.setObjectPosition(droneHandle, -1, [0.0, 0.0, 1.0])
    
    # Start simulation in stepping mode.
    sim.setStepping(True)
    sim.startSimulation()
    
    # Model parameters.
    dt = sim.getSimulationTimeStep()  # Time step [s]
    N = 10                            # MPC prediction horizon
    m_drone = 0.52                    # Drone mass [kg]
    g = 9.81                          # Gravitational acceleration [m/s²]
    l_arm = 0.13                      # Arm length for force allocation [m]
    
    # Continuous-time system matrices.
    A = np.zeros((12, 12))
    A[0, 3] = 1; A[1, 4] = 1; A[2, 5] = 1  # Position integration.
    A[3, 7] = g; A[4, 6] = -g               # Linearized translational dynamics.
    A[6, 9] = 1; A[7, 10] = 1; A[8, 11] = 1   # Attitude kinematics.
    B = np.zeros((12, 4))
    B[5, 0] = 1.0 / m_drone  # z acceleration from thrust deviation.
    B[9, 1] = 1.0            # roll acceleration.
    B[10, 2] = 1.0           # pitch acceleration.
    B[11, 3] = 1.0           # yaw acceleration.
    
    # Discretize system (Euler discretization).
    A_d = np.eye(12) + A * dt
    B_d = B * dt
    R = np.diag([0.1, 0.4, 0.4, 0.1])
    u_min = np.array([-m_drone * g, -0.015, -0.015, -1.0])
    u_max = np.array([1.0, 0.015, 0.015, 1.0])
    
    # Create a PID controller instance.
    pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)
    
    # Simulation loop.
    while (t := sim.getSimulationTime()) < simualtion_time:
        # State estimation.
        pos = sim.getObjectPosition(droneHandle, -1)
        linVel, _ = sim.getObjectVelocity(droneHandle)
        ori = sim.getObjectOrientation(droneHandle, -1)
        _, angVel = sim.getObjectVelocity(droneHandle)
        x0 = np.array([pos[0], pos[1], pos[2],
                       linVel[0], linVel[1], linVel[2],
                       ori[0], ori[1], ori[2],
                       angVel[0], angVel[1], angVel[2]])
        time_log.append(t)
        state_log.append(x0)
        
        # Update target position (example: step update).
        if t >= 0.1 and t < 5:
            sim.setObjectPosition(targetHandle, -1, [2.0, 2.0, 3.0])
        targetPos = sim.getObjectPosition(targetHandle, -1)
        target_log.append(targetPos)
        ref = np.array([targetPos[0], targetPos[1], targetPos[2],
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0])
        
        # Compute distance error.
        x_err = targetPos[0] - pos[0]
        y_err = targetPos[1] - pos[1]
        z_err = targetPos[2] - pos[2]
        distance = np.sqrt(x_err**2 + y_err**2 + z_err**2)
        distance_log.append(distance)
        
        # Solve the MPC problem.
        try:
            u_opt, x_pred = mpc_controller(
                x0, A_d, B_d, ref, N, Q, R, u_min, u_max, pid_controller=pid_z
            )
        except Exception as e:
            print("MPC error:", e)
            u_opt = np.zeros(4)
        control_log.append(u_opt)
        
        # Compute thrust with gravitational compensation.
        total_thrust = u_opt[0] + m_drone * g
        total_thrust = min(total_thrust, m_drone * g * 1.5)
        u_fin = [total_thrust, u_opt[1], u_opt[2], u_opt[3]]
        
        # Force/Torque allocation.
        f0 = total_thrust / 4 + u_opt[1] / (2 * l_arm) - u_opt[2] / (2 * l_arm) - u_opt[3] / 4
        f1 = total_thrust / 4 + u_opt[1] / (2 * l_arm) + u_opt[2] / (2 * l_arm) + u_opt[3] / 4
        f2 = total_thrust / 4 - u_opt[1] / (2 * l_arm) + u_opt[2] / (2 * l_arm) - u_opt[3] / 4
        f3 = total_thrust / 4 - u_opt[1] / (2 * l_arm) - u_opt[2] / (2 * l_arm) + u_opt[3] / 4
        force_log.append([f0, f1, f2, f3])
        
        forces_body = [[0, 0, f0],
                       [0, 0, f1],
                       [0, 0, f2],
                       [0, 0, f3]]
        bodyMatrix = sim.getObjectMatrix(droneHandle, -1)
        bodyMatrix[3] = 0; bodyMatrix[7] = 0; bodyMatrix[11] = 0
        forces_world = [sim.multiplyVector(bodyMatrix, force) for force in forces_body]
        
        for i in range(4):
            sim.addForceAndTorque(propellerHandle[i], forces_world[i], [0, 0, 0])
        
        # Rotate propellers (visual effect).
        sim.setJointTargetVelocity(jointHandle[0], -100)
        sim.setJointTargetVelocity(jointHandle[1], 100)
        sim.setJointTargetVelocity(jointHandle[2], -100)
        sim.setJointTargetVelocity(jointHandle[3], 100)
        
        # Step simulation.
        sim.step()
    
    sim.stopSimulation()
    cost = sum([d**2 for d in distance_log])
    return cost

# --- Setup for optimization ---
# Full Q diagonal initial guess (12 elements)
initial_guess_full = np.array([0.3, 0.3, 3.0, 0.01, 0.01, 0.1, 0.25, 0.25, 0.1, 0.0, 0.0, 0.0])
# Free indices: optimize elements 0, 1, 2 and 6, 7, 8.
free_indices = [0, 1, 2, 6, 7, 8]

def cost_function_free(free_params):
    # Construct full Q vector: update free indices, keep others fixed.
    full_Q_params = initial_guess_full.copy()
    full_Q_params[free_indices] = free_params
    Q_matrix = np.diag(full_Q_params)
    cost = run_simulation(Q_matrix)
    print("Trying free parameters:", free_params, "-> Full Q:", full_Q_params, "=> Cost:", cost)
    return cost

# Set bounds to ensure free parameters stay nonnegative.
bounds = [(1e-6, None)] * len(free_indices)

# Free parameters initial guess (from initial_guess_full at free indices)
free_initial = initial_guess_full[free_indices]

# Optimize using scipy.optimize.minimize with L-BFGS-B
result = minimize(cost_function_free, free_initial, method='L-BFGS-B', bounds=bounds)

optimized_free = result.x
optimized_full = initial_guess_full.copy()
optimized_full[free_indices] = optimized_free

print("Optimized Q parameters (full vector):", optimized_full)
