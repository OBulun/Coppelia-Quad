#!/usr/bin/env python3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    # Load the simulation log CSV
    df = pd.read_csv('./simulation_logs.csv')
    filename = "./logs/simulation_logs_latest.html"
    # Create a 3×2 grid of subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Positions vs Time', 'Velocities vs Time',
            'Euler Angles vs Time', 'Control Inputs vs Time',
            'Propeller Forces vs Time', 'Distance to Target vs Time'
        ]
    )

    # --- Positions vs Time (row=1, col=1) ---
    for axis in ['x', 'y', 'z']:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', name=axis),
            row=1, col=1
        )
    for axis in ['target_x', 'target_y', 'target_z']:
        name = axis.replace('target_', 'target ')
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', line=dict(dash='dash'), name=name),
            row=1, col=1
        )
    fig.update_xaxes(title_text='Time [s]', row=1, col=1)
    fig.update_yaxes(title_text='Position [m]', row=1, col=1)

    # --- Velocities vs Time (row=1, col=2) ---
    for axis in ['vx', 'vy', 'vz']:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', name=axis),
            row=1, col=2
        )
    fig.update_xaxes(title_text='Time [s]', row=1, col=2)
    fig.update_yaxes(title_text='Velocity [m/s]', row=1, col=2)

    # --- Euler Angles vs Time (row=2, col=1) ---
    for axis in ['roll', 'pitch', 'yaw']:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', name=axis),
            row=2, col=1
        )
    fig.update_xaxes(title_text='Time [s]', row=2, col=1)
    fig.update_yaxes(title_text='Angle [rad]', row=2, col=1)

    # --- Control Inputs vs Time (row=2, col=2) ---
    for axis in ['delta_thrust', 'roll_torque', 'pitch_torque', 'yaw_torque']:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', name=axis),
            row=2, col=2
        )
    fig.update_xaxes(title_text='Time [s]', row=2, col=2)
    fig.update_yaxes(title_text='Control Input', row=2, col=2)

    # --- Propeller Forces vs Time (row=3, col=1) ---
    for axis in ['f0', 'f1', 'f2', 'f3']:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[axis], mode='lines', name=axis),
            row=3, col=1
        )
    fig.update_xaxes(title_text='Time [s]', row=3, col=1)
    fig.update_yaxes(title_text='Force [N]', row=3, col=1)

    # --- Distance to Target vs Time (row=3, col=2) ---
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['distance'], mode='lines', name='distance'),
        row=3, col=2
    )
    fig.update_xaxes(title_text='Time [s]', row=3, col=2)
    fig.update_yaxes(title_text='Distance [m]', row=3, col=2)

    # Final layout tweaks
    fig.update_layout(
        height=2560, width=1440,
        title_text='Simulation Logs Overview',
        legend=dict(traceorder='grouped')
    )

    # Save to HTML for interactive view
    fig.write_html(filename, include_plotlyjs='cdn')
    print(f"Interactive plot saved to {filename}")


if __name__ == '__main__':
    main()
