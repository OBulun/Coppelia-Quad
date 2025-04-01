import numpy as np

def generate_trajectory(p0, p1, total_time, dt, v0=None, v1=None):
    # Generates a smooth trajectory from an initial position p0 to a target position p1 using cubic polynomial interpolation.
    # p0: Initial position (3D vector: [x, y, z])
    # p1: Target position (3D vector: [x, y, z])
    # total_time: Total duration of the segment [s]
    # dt: Time step [s]
    # v0: Optional initial velocity (defaults to zero if None)
    # v1: Optional final velocity (defaults to zero if None)
    #
    # Returns:
    # t: Time vector for the segment
    # positions: Array of positions along the segment (shape: [len(t), 3])
    # velocities: Array of velocities along the segment (shape: [len(t), 3])
    # accelerations: Array of accelerations along the segment (shape: [len(t), 3])
    
    if v0 is None:
        v0 = np.zeros_like(p0)
    if v1 is None:
        v1 = np.zeros_like(p1)
    
    t = np.arange(0, total_time + dt, dt)
    num_steps = len(t)
    positions = np.zeros((num_steps, len(p0)))
    velocities = np.zeros((num_steps, len(p0)))
    accelerations = np.zeros((num_steps, len(p0)))
    
    for i in range(len(p0)):
        a0 = p0[i]
        a1 = v0[i]
        a2 = (3 * (p1[i] - p0[i]) - (2 * v0[i] + v1[i]) * total_time) / (total_time ** 2)
        a3 = (-2 * (p1[i] - p0[i]) + (v0[i] + v1[i]) * total_time) / (total_time ** 3)
        
        positions[:, i] = a0 + a1 * t + a2 * t**2 + a3 * t**3
        velocities[:, i] = a1 + 2 * a2 * t + 3 * a3 * t**2
        accelerations[:, i] = 2 * a2 + 6 * a3 * t
        
    return t, positions, velocities, accelerations

def generate_landmower_pattern(x_min, x_max, y_min, y_max, row_spacing, points_per_row):
    # Generates a landmower (boustrophedon) pattern as a list of (x,y) tuples.
    # x_min, x_max: boundaries in the x direction
    # y_min, y_max: boundaries in the y direction
    # row_spacing: spacing between consecutive rows in y
    # points_per_row: number of points along x for each row
    y_positions = np.arange(y_min, y_max + row_spacing, row_spacing)
    waypoints = []
    
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            # Even row: x from x_min to x_max
            x_positions = np.linspace(x_min, x_max, points_per_row)
        else:
            # Odd row: x from x_max to x_min (reverse order)
            x_positions = np.linspace(x_max, x_min, points_per_row)
        for x in x_positions:
            waypoints.append((x, y))
    
    return waypoints

def generate_landmower_trajectory(x_min, x_max, y_min, y_max, row_spacing, points_per_row, segment_time, dt, z_const):
    # Creates a landmower pattern trajectory and returns it in the same format as generate_trajectory.
    # The area in the x-y plane is defined by x_min, x_max, y_min, y_max.
    # row_spacing and points_per_row define the pattern.
    # Each segment connecting consecutive waypoints will take segment_time seconds.
    # dt is the time step for trajectory sampling.
    # z_const is the constant z-coordinate for all waypoints.
    
    # Generate 2D waypoints and add constant z to get full 3D points.
    pattern = generate_landmower_pattern(x_min, x_max, y_min, y_max, row_spacing, points_per_row)
    waypoints = [np.array([pt[0], pt[1], z_const]) for pt in pattern]
    
    global_t = []
    global_positions = []
    global_velocities = []
    global_accelerations = []
    
    time_offset = 0  # To ensure the global time vector is continuous
    
    # Generate a trajectory segment between each pair of consecutive waypoints.
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i+1]
        t_seg, pos_seg, vel_seg, acc_seg = generate_trajectory(p0, p1, segment_time, dt)
        
        # Offset the time vector so segments connect continuously.
        t_seg = t_seg + time_offset
        
        # Avoid duplicating points between segments (skip the first point for all but the first segment).
        if i > 0:
            t_seg = t_seg[1:]
            pos_seg = pos_seg[1:]
            vel_seg = vel_seg[1:]
            acc_seg = acc_seg[1:]
        
        global_t.append(t_seg)
        global_positions.append(pos_seg)
        global_velocities.append(vel_seg)
        global_accelerations.append(acc_seg)
        
        time_offset = global_t[-1][-1]
    
    # Concatenate all segments into single arrays.
    global_t = np.concatenate(global_t)
    global_positions = np.concatenate(global_positions)
    global_velocities = np.concatenate(global_velocities)
    global_accelerations = np.concatenate(global_accelerations)
    
    return global_t, global_positions, global_velocities, global_accelerations

# Example usage:
if __name__ == "__main__":
    # Define the area boundaries for x and y.
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    
    # Parameters for the landmower pattern.
    row_spacing = 0.5         # Spacing between rows in y
    points_per_row = 10       # Number of waypoints per row
    
    # Trajectory parameters for each segment connecting waypoints.
    segment_time = 2.0        # Duration [s] for each segment
    dt = 0.01                 # Time step for trajectory sampling
    z_const = 1.0             # Constant altitude for the trajectory
    
    # Generate the complete landmower trajectory.
    t, positions, velocities, accelerations = generate_landmower_trajectory(
        x_min, x_max, y_min, y_max, row_spacing, points_per_row, segment_time, dt, z_const
    )
    
    print("Final position:", positions[-1])
