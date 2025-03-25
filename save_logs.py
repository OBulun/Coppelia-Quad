import pandas as pd

def save_logs(time_log, state_log, control_log, force_log, target_log, filename="simulation_logs.csv"):
    # Create a DataFrame with appropriate column names.
    df = pd.DataFrame({
        "time": time_log,
        "x": [s[0] for s in state_log],
        "y": [s[1] for s in state_log],
        "z": [s[2] for s in state_log],
        "vx": [s[3] for s in state_log],
        "vy": [s[4] for s in state_log],
        "vz": [s[5] for s in state_log],
        "roll": [s[6] for s in state_log],
        "pitch": [s[7] for s in state_log],
        "yaw": [s[8] for s in state_log],
        "p": [s[9] for s in state_log],
        "q": [s[10] for s in state_log],
        "r": [s[11] for s in state_log],
        "delta_thrust": [u[0] for u in control_log],
        "roll_torque": [u[1] for u in control_log],
        "pitch_torque": [u[2] for u in control_log],
        "yaw_torque": [u[3] for u in control_log],
        "f0": [f[0] for f in force_log],
        "f1": [f[1] for f in force_log],
        "f2": [f[2] for f in force_log],
        "f3": [f[3] for f in force_log],
        "target_x": [t[0] for t in target_log],
        "target_y": [t[1] for t in target_log],
        "target_z": [t[2] for t in target_log],
    })
    df.to_csv(filename, index=False)
    print(f"Logs saved to {filename}")
