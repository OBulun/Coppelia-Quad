import pandas as pd
import matplotlib.pyplot as plt

def read_params(filename):
    with open(filename, "r") as f:
        return f.read()

def main():
    # Read the simulation logs CSV.
    df = pd.read_csv("simulation_logs.csv")
    
    # Read the parameter file.
    params_text = read_params("simulation_parameters.txt")
    
    # Create a figure with subplots.
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # --- Plot Positions ---
    axs[0, 0].plot(df["time"], df["x"], label="x")
    axs[0, 0].plot(df["time"], df["y"], label="y")
    axs[0, 0].plot(df["time"], df["z"], label="z")
    # Plot target positions with dashed lines.
    axs[0, 0].plot(df["time"], df["target_x"], label="target x", linestyle="--")
    axs[0, 0].plot(df["time"], df["target_y"], label="target y", linestyle="--")
    axs[0, 0].plot(df["time"], df["target_z"], label="target z", linestyle="--")
    axs[0, 0].set_title("Positions vs Time")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Position [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # --- Plot Velocities ---
    axs[0, 1].plot(df["time"], df["vx"], label="vx")
    axs[0, 1].plot(df["time"], df["vy"], label="vy")
    axs[0, 1].plot(df["time"], df["vz"], label="vz")
    axs[0, 1].set_title("Velocities vs Time")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Velocity [m/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # --- Plot Euler Angles ---
    axs[1, 0].plot(df["time"], df["roll"], label="roll")
    axs[1, 0].plot(df["time"], df["pitch"], label="pitch")
    axs[1, 0].plot(df["time"], df["yaw"], label="yaw")
    axs[1, 0].set_title("Euler Angles vs Time")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Angle [rad]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # --- Plot Control Inputs ---
    axs[1, 1].plot(df["time"], df["delta_thrust"], label="delta_thrust")
    axs[1, 1].plot(df["time"], df["roll_torque"], label="roll_torque")
    axs[1, 1].plot(df["time"], df["pitch_torque"], label="pitch_torque")
    axs[1, 1].plot(df["time"], df["yaw_torque"], label="yaw_torque")
    axs[1, 1].set_title("Control Inputs vs Time")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Control Input")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # --- Plot Forces (f0, f1, f2, f3) ---
    axs[2, 0].plot(df["time"], df["f0"], label="f0")
    axs[2, 0].plot(df["time"], df["f1"], label="f1")
    axs[2, 0].plot(df["time"], df["f2"], label="f2")
    axs[2, 0].plot(df["time"], df["f3"], label="f3")
    axs[2, 0].set_title("Propeller Forces vs Time")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Force [N]")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # --- Unused subplot (bottom-right) ---
    axs[2, 1].axis("off")
    
    # Add a text annotation with the parameter information.
    # Here we position it at (0.9, 0.01); adjust as needed.
    fig.text(0.9, 0.01, params_text, ha="center", fontsize=8, wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the annotation.
    plt.show()

if __name__ == "__main__":
    main()
