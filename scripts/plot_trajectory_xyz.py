import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Load the simulation logs
log_filename = "simulation_logs.csv"
df = pd.read_csv(log_filename)

# Extract positions for drone and target (x, y, z)
drone_x = df["x"].values
drone_y = df["y"].values
drone_z = df["z"].values
target_x = df["target_x"].values
target_y = df["target_y"].values
target_z = df["target_z"].values

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("Drone and Target 3D Trajectories")

# Determine plot limits with some padding
all_x = np.concatenate([drone_x, target_x])
all_y = np.concatenate([drone_y, target_y])
all_z = np.concatenate([drone_z, target_z])
padding = 1.0
ax.set_xlim(np.min(all_x) - padding, np.max(all_x) + padding)
ax.set_ylim(np.min(all_y) - padding, np.max(all_y) + padding)
ax.set_zlim(np.min(all_z) - padding, np.max(all_z) + padding)

# Initialize plot elements: trajectories and current position markers.
# For 3D plots, plot returns a list of line objects so we unpack them.
drone_traj_line, = ax.plot([], [], [], 'b-', linewidth=2, label="Drone Trajectory")
target_traj_line, = ax.plot([], [], [], 'r--', linewidth=2, label="Target Trajectory")
drone_dot, = ax.plot([], [], [], 'bo', markersize=8)
target_dot, = ax.plot([], [], [], 'ro', markersize=8)

ax.legend(loc="upper right")

def init():
    drone_traj_line.set_data([], [])
    drone_traj_line.set_3d_properties([])
    target_traj_line.set_data([], [])
    target_traj_line.set_3d_properties([])
    drone_dot.set_data([], [])
    drone_dot.set_3d_properties([])
    target_dot.set_data([], [])
    target_dot.set_3d_properties([])
    return drone_traj_line, target_traj_line, drone_dot, target_dot

def animate(i):
    # Update the trajectory lines
    drone_traj_line.set_data(drone_x[:i+1], drone_y[:i+1])
    drone_traj_line.set_3d_properties(drone_z[:i+1])
    
    target_traj_line.set_data(target_x[:i+1], target_y[:i+1])
    target_traj_line.set_3d_properties(target_z[:i+1])
    
    # Update current position markers (wrap scalars in lists)
    drone_dot.set_data([drone_x[i]], [drone_y[i]])
    drone_dot.set_3d_properties([drone_z[i]])
    
    target_dot.set_data([target_x[i]], [target_y[i]])
    target_dot.set_3d_properties([target_z[i]])
    
    return drone_traj_line, target_traj_line, drone_dot, target_dot

# Create the animation (blitting is generally not supported in 3D)
num_frames = len(drone_x)
ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                              interval=0.01, blit=False)

plt.show()
