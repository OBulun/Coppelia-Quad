#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import noise  # Import noise functions from noise.py

class Drone:
    def __init__(self, m=1.0, g=9.81, Ixx=0.005, Iyy=0.005, Izz=0.009, l=0.1, k_m=0.01):
        """
        Initialize the drone with physical parameters.
        m    : Mass (kg)
        g    : Gravity (m/s^2)
        Ixx  : Moment of inertia about x-axis (kg·m^2)
        Iyy  : Moment of inertia about y-axis (kg·m^2)
        Izz  : Moment of inertia about z-axis (kg·m^2)
        l    : Distance from the center to each motor (m)
        k_m  : Motor moment constant for yaw
        """
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.l = l
        self.k_m = k_m

        # Initial state (12-dimensional): [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.initial_state = np.zeros(12)
        self.initial_state[2] = 0.0  # Starting altitude (m)
        # Extra force disturbance (default is zero force)
        self.extra_force = np.zeros(3)

    def dynamics(self, t, state, motor_inputs):
        """
        Compute the time derivative of the state vector.
        
        Parameters:
          t            : Time (s)
          state        : 12-element state vector.
          motor_inputs : List or array of four motor thrusts [u1, u2, u3, u4] (N)
          
        Returns:
          state_dot    : Time derivative of the state vector.
        """
        # Unpack state
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        # Unpack motor inputs
        u1, u2, u3, u4 = motor_inputs

        # Total thrust from all motors
        total_thrust = u1 + u2 + u3 + u4

        # Precompute trigonometric functions
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Rotation matrix from body frame to inertial frame (Z-up convention)
        R = np.array([
            [ctheta * cpsi, ctheta * spsi, -stheta],
            [sphi * stheta * cpsi - cphi * spsi, sphi * stheta * spsi + cphi * cpsi, sphi * ctheta],
            [cphi * stheta * cpsi + sphi * spsi, cphi * stheta * spsi - sphi * cpsi, cphi * ctheta]
        ])

        # Translational acceleration in the inertial frame
        thrust_body = np.array([0, 0, total_thrust])
        accel = (R @ thrust_body) / self.m - np.array([0, 0, self.g])
        # Include any extra force disturbance (e.g., wind or extra weight effects)
        if hasattr(self, 'extra_force'):
            accel += self.extra_force / self.m

        # Moments (assuming a cross configuration)
        L = self.l * (u4 - u2)            # Roll moment
        M = self.l * (u3 - u1)            # Pitch moment
        N = self.k_m * (u1 - u2 + u3 - u4)  # Yaw moment

        # Angular accelerations using Euler's equations
        p_dot = (L + (self.Iyy - self.Izz) * q * r) / self.Ixx
        q_dot = (M + (self.Izz - self.Ixx) * p * r) / self.Iyy
        r_dot = (N + (self.Ixx - self.Iyy) * p * q) / self.Izz

        # Euler angle derivatives from angular velocities
        phi_dot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        theta_dot = np.cos(phi) * q - np.sin(phi) * r
        psi_dot = (np.sin(phi) / ctheta) * q + (np.cos(phi) / ctheta) * r

        # Assemble the state derivative vector
        state_dot = np.zeros(12)
        state_dot[0:3] = [vx, vy, vz]      # Derivatives of position
        state_dot[3:6] = accel             # Derivatives of velocity
        state_dot[6] = phi_dot             # Derivatives of Euler angles
        state_dot[7] = theta_dot
        state_dot[8] = psi_dot
        state_dot[9:12] = [p_dot, q_dot, r_dot]  # Angular accelerations

        return state_dot

def simulate_drone(drone, motor_inputs=None, t_span=(0, 5), num_points=501, modeling_noise_std=0.0):
    """
    Simulate the drone dynamics.
    
    Parameters:
      drone             : Instance of the Drone class.
      motor_inputs      : Optional list of four motor thrusts. If None, uses hover thrust.
      t_span            : Tuple with start and end times.
      num_points        : Number of time points for evaluation.
      modeling_noise_std: Standard deviation for modeling noise (default is 0.0, i.e. no noise).
      
    Returns:
      solution          : The solution object from solve_ivp.
    """
    # Default hover condition: each motor provides mg/4 thrust
    if motor_inputs is None:
        hover_thrust = drone.m * drone.g / 4.0
        motor_inputs = [hover_thrust] * 4

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    def dynamics_with_noise(t, s):
        state_dot = drone.dynamics(t, s, motor_inputs)
        if modeling_noise_std > 0.0:
            state_dot = noise.add_modeling_noise(state_dot, modeling_noise_std)
        return state_dot

    solution = solve_ivp(dynamics_with_noise, t_span, drone.initial_state, t_eval=t_eval, rtol=1e-6)
    return solution

def plot_simulation(solution):
    """
    Plot the simulation results.
    
    Parameters:
      solution : The solution object returned by solve_ivp.
    """
    plt.figure(figsize=(10, 8))

    # Plot positions: x, y, z
    plt.subplot(3, 1, 1)
    plt.plot(solution.t, solution.y[0], label='x')
    plt.plot(solution.t, solution.y[1], label='y')
    plt.plot(solution.t, solution.y[2], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()

    # Plot orientations (Euler angles in degrees)
    plt.subplot(3, 1, 2)
    plt.plot(solution.t, solution.y[6] * 180/np.pi, label='Roll (φ)')
    plt.plot(solution.t, solution.y[7] * 180/np.pi, label='Pitch (θ)')
    plt.plot(solution.t, solution.y[8] * 180/np.pi, label='Yaw (ψ)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (°)')
    plt.title('Orientation vs Time')
    plt.legend()

    # Plot angular velocities: p, q, r
    plt.subplot(3, 1, 3)
    plt.plot(solution.t, solution.y[9], label='p')
    plt.plot(solution.t, solution.y[10], label='q')
    plt.plot(solution.t, solution.y[11], label='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocities vs Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_final_state(solution, measurement_noise_std=0.0):
    """
    Print the final state of the simulation.
    Optionally add measurement noise to simulate sensor readings.
    
    Parameters:
      solution             : The solution object returned by solve_ivp.
      measurement_noise_std: Standard deviation for measurement noise.
    """
    final_state = solution.y[:, -1]
    if measurement_noise_std > 0.0:
        final_state = noise.add_measurement_noise(final_state, measurement_noise_std)
    print("\nFinal state vector:")
    print("Position (x, y, z):", final_state[0:3])
    print("Velocity (vx, vy, vz):", final_state[3:6])
    print("Orientation (phi, theta, psi) in radians:", final_state[6:9])
    print("Angular velocities (p, q, r):", final_state[9:12])
