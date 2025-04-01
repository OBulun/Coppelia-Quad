import numpy as np

def dynamic_window_approach(current_state, target_state, dynamic_constraints, dt, horizon):
    """
    Compute a reference state for the MPC by evaluating candidate acceleration inputs
    using a simplified double-integrator model. The reference state includes position,
    velocity, desired roll, pitch, yaw, and their corresponding angular rates (p, q, r).

    Dynamic constraints include:
      - "v_max": Maximum linear velocities (np.array([vx_max, vy_max, vz_max])).
      - "a_max": Maximum linear accelerations (np.array([ax_max, ay_max, az_max])).
      - "angle_max": Maximum attitude angles (np.array([max_roll, max_pitch, max_yaw])) in radians.
      - "angular_rate_max": Maximum angular rates (np.array([max_roll_rate, max_pitch_rate, max_yaw_rate])) in rad/s.
      - "num_samples": Number of candidate samples per axis.

    Parameters:
      current_state      : numpy array (12-dimensional state vector)
                           [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
      target_state       : numpy array (12-dimensional state vector)
                           (Only the target position is used; the rest can be zeros)
      dynamic_constraints: dict containing the dynamic and attitude constraints.
      dt                 : float, time step for simulation.
      horizon            : int, number of simulation steps (planning horizon).

    Returns:
      ref_state          : numpy array (12-dimensional reference state)
                           [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    """
    # Extract current position and velocity.
    current_pos = current_state[0:3]
    current_vel = current_state[3:6]
    
    # Linear dynamic constraints.
    v_max = dynamic_constraints.get("v_max", np.array([1.0, 1.0, 1.0]))
    a_max = dynamic_constraints.get("a_max", np.array([0.5, 0.5, 0.5]))
    num_samples = dynamic_constraints.get("num_samples", 5)
    
    # Attitude constraints.
    angle_max = dynamic_constraints.get("angle_max", np.array([np.radians(30), np.radians(30), np.radians(30)]))
    
    
    # Generate candidate accelerations for each axis.
    a_x_candidates = np.linspace(-a_max[0], a_max[0], num_samples)
    a_y_candidates = np.linspace(-a_max[1], a_max[1], num_samples)
    a_z_candidates = np.linspace(-a_max[2], a_max[2], num_samples)
    
    best_cost = float("inf")
    best_final_pos = None
    best_final_vel = None
    best_a_candidate = None

    # Cost weights (tune these parameters as needed)
    w_pos = 2.0   # Weight for position error.
    w_vel = 0.2   # Weight for residual velocity.
    w_acc = 0.01  # Weight for control effort.
    
    # Evaluate each candidate acceleration combination.
    for ax in a_x_candidates:
        for ay in a_y_candidates:
            for az in a_z_candidates:
                a_candidate = np.array([ax, ay, az])
                pos = current_pos.copy()
                vel = current_vel.copy()
                # Simulate the trajectory over the planning horizon.
                for _ in range(horizon):
                    vel = vel + a_candidate * dt
                    vel = np.clip(vel, -v_max, v_max)
                    pos = pos + vel * dt + 0.5 * a_candidate * (dt ** 2)
                
                # Calculate cost (position error, velocity error, and control effort).
                cost_pos = np.linalg.norm(pos - target_state[0:3])**2
                cost_vel = np.linalg.norm(vel)**2
                cost_acc = np.linalg.norm(a_candidate)**2
                cost = w_pos * cost_pos + w_vel * cost_vel + w_acc * cost_acc
                
                if cost < best_cost:
                    best_cost = cost
                    best_final_pos = pos
                    best_final_vel = vel
                    best_a_candidate = a_candidate

    # Convert the best candidate acceleration into desired attitude commands.
    # Using a simplified model:
    g = 9.81  # Gravitational acceleration.
    desired_pitch = np.arctan2(best_a_candidate[0], g)  # To achieve acceleration in x.
    desired_roll  = -np.arctan2(best_a_candidate[1], g) # To achieve acceleration in y.
    desired_yaw   = 0.0  # Fixed or computed separately.

    # Apply constraints to the attitude.
    desired_roll  = np.clip(desired_roll, -angle_max[0], angle_max[0])
    desired_pitch = np.clip(desired_pitch, -angle_max[1], angle_max[1])
    desired_yaw   = np.clip(desired_yaw, -angle_max[2], angle_max[2])
    
    # Compute the angular rates required to achieve the desired attitude.
    time_horizon = horizon * dt
    current_roll  = current_state[6]
    current_pitch = current_state[7]
    current_yaw   = current_state[8]
    
    desired_roll_rate  = (desired_roll - current_roll) / time_horizon
    desired_pitch_rate = (desired_pitch - current_pitch) / time_horizon
    desired_yaw_rate   = (desired_yaw - current_yaw) / time_horizon
    

    
    
    # Build the full 12-dimensional reference state.
    ref_state = np.zeros(12)
    ref_state[0:3]  = best_final_pos       # Position.
    ref_state[3:6]  = best_final_vel       # Velocity.
    ref_state[6]    = desired_roll         # Roll.
    ref_state[7]    = desired_pitch        # Pitch.
    ref_state[8]    = desired_yaw          # Yaw.
    ref_state[9]    = desired_roll_rate    # Roll rate (p).
    ref_state[10]   = desired_pitch_rate   # Pitch rate (q).
    ref_state[11]   = desired_yaw_rate     # Yaw rate (r).
    
    return ref_state


