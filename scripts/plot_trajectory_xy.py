import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load the simulation logs
log_filename = "simulation_logs.csv"
df = pd.read_csv(log_filename)

# Extract positions for drone and target
drone_x = df["x"].values
drone_y = df["y"].values
target_x = df["target_x"].values
target_y = df["target_y"].values

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Drone and Target Trajectories")

# Determine plot limits with some padding
all_x = np.concatenate([drone_x, target_x])
all_y = np.concatenate([drone_y, target_y])
padding = 1.0
ax.set_xlim(np.min(all_x) - padding, np.max(all_x) + padding)
ax.set_ylim(np.min(all_y) - padding, np.max(all_y) + padding)

# Initialize plot elements: trajectories and current positions
drone_traj_line, = ax.plot([], [], 'b-', linewidth=2, label="Drone Trajectory")
target_traj_line, = ax.plot([], [], 'r--', linewidth=2, label="Target Trajectory")
drone_dot, = ax.plot([], [], 'bo', markersize=8)
target_dot, = ax.plot([], [], 'ro', markersize=8)

ax.legend(loc="upper right")

def init():
    drone_traj_line.set_data([], [])
    target_traj_line.set_data([], [])
    drone_dot.set_data([], [])
    target_dot.set_data([], [])
    return drone_traj_line, target_traj_line, drone_dot, target_dot

def animate(i):
    # Update trajectory lines and current position markers
    drone_traj_line.set_data(drone_x[:i+1], drone_y[:i+1])
    target_traj_line.set_data(target_x[:i+1], target_y[:i+1])
    drone_dot.set_data([drone_x[i]], [drone_y[i]])
    target_dot.set_data([target_x[i]], [target_y[i]])
    return drone_traj_line, target_traj_line, drone_dot, target_dot

# Create the animation
num_frames = len(drone_x)
ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                              interval=0.01, blit=True)

plt.show()
