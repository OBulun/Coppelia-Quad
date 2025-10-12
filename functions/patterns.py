"""
Module: patterns.py

Provides a PatternGenerator class to produce target trajectories for a quadcopter.
Usage:
    from patterns import PatternGenerator
    gen = PatternGenerator(pattern_id, sim_time)
    target_pos = gen.get_position(t)
"""

import numpy as np
import random

class PatternGenerator:
    def __init__(self, pattern, sim_time):
        """
        pattern: int code for which trajectory to generate
        sim_time: total simulation time [s]
        """
        self.pattern = pattern
        self.sim_time = sim_time
        # Constants
        self.R_circle     = 3.0
        self.omega_circle = 0.5
        self.z_fixed      = 1.5
        self.A_vert       = 0.5
        self.omega_vert   = 1.0
        # new speed factor for Lissajous (pattern 12)
        self.lissajous_speed = 0.2  # <1 slows down, ~1 default
        # Random waypoints setup
        if pattern == 7:
            random.seed(42)
            self.num_wp = 5
            self.waypoints = [
                np.array([random.uniform(-3,3), random.uniform(-3,3), random.uniform(1,2)])
                for _ in range(self.num_wp)
            ]
            self.wp_dur = sim_time / self.num_wp

    def get_position(self, t):
        """
        Return [x, y, z] at time t for the selected pattern.
        """
        p = self.pattern
        # --- Pattern 0: Christmas tree spiral with rising altitude ---
        if p == 0:
            end_hold = 10.0
            if self.sim_time - t < end_hold:
                z = 1.0 + 0.1 * (self.sim_time - end_hold)
                return [0.0, 0.0, z]
            r = 4 * np.exp(-0.05 * t)
            x = r * np.sin(0.9 * t)
            y = r * np.cos(0.9 * t)
            z = 1.0 + 0.1 * t
            return [x, y, z]

        # --- Pattern 1: Rectangular step ---
        elif p == 1:
            z = 2.0
            if 0.0 <= t < 5.0:
                return [ 2.0,  2.0, z]
            elif 5.0 <= t < 10.0:
                return [-2.0,  2.0, z]
            elif 10.0 <= t < 15.0:
                return [-2.0, -2.0, z]
            elif 15.0 <= t < 20.0:
                return [ 2.0, -2.0, z]
            else:
                return [ 0.0,  0.0, z]

        # --- Pattern 2: Circle ---
        elif p == 2:
            x = self.R_circle * np.cos(self.omega_circle * t)
            y = self.R_circle * np.sin(self.omega_circle * t)
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 3: Figure-eight (lemniscate-like) ---
        elif p == 3:
            R = self.R_circle
            w = self.omega_circle
            x = R * np.sin(w * t)
            y = R * np.sin(w * t) * np.cos(w * t)
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 4: Helix ---
        elif p == 4:
            R = 2.0
            w = self.omega_circle
            climb = 0.05
            x = R * np.cos(w * t)
            y = R * np.sin(w * t)
            z = 1.0 + climb * t
            return [x, y, z]

        # --- Pattern 5: Lawnmower sweep ---
        elif p == 5:
            Xmax = 3.0
            Ymax = 3.0
            strips = 4
            dt = self.sim_time / strips
            idx = int(min(t // dt, strips-1))
            y = -Ymax + (2*Ymax/(strips-1)) * idx
            x =  Xmax if idx % 2 == 0 else -Xmax
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 6: S-curve ---
        elif p == 6:
            Xmax = 3.0
            A = 2.0
            B = 1.0
            x = -Xmax + (2*Xmax/self.sim_time) * t
            y = A * np.tanh(B * (t - self.sim_time/2))
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 7: Random waypoints ---
        elif p == 7:
            idx = int(min(t // self.wp_dur, self.num_wp-1))
            return self.waypoints[idx].tolist()

        # --- Pattern 8: Vertical oscillation ---
        elif p == 8:
            z = self.z_fixed + self.A_vert * np.sin(self.omega_vert * t)
            return [0.0, 0.0, z]

        # --- Pattern 9: Polygon (hexagon) ---
        elif p == 9:
            sides = 6
            R = 3.0
            dt = self.sim_time / sides
            idx = int(min(t // dt, sides-1))
            theta = 2*np.pi * idx / sides
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 10: Lemniscate of Bernoulli ---
        elif p == 10:
            R = 3.0
            w = self.omega_circle
            sint = np.sin(w*t)
            cost = np.cos(w*t)
            denom = 1 + cost**2
            x = R * sint / denom
            y = R * sint * cost / denom
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 11: Ramp and hold ---
        elif p == 11:
            Xmax = 3.0
            third = self.sim_time / 3
            if t < third:
                x = -Xmax + (2*Xmax/third) * t
            elif t < 2*third:
                x = Xmax
            else:
                x = Xmax - (2*Xmax/third) * (t-2*third)
            y = 0.0
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 12: 3D Lissajous curve (slowed down) ---
        elif p == 12:
            # apply speed factor to time
            tau = self.lissajous_speed * t
            A, B, C = 3.0, 3.0, 1.0
            a, b, c = 3.0, 4.0, 2.0
            delta = np.pi/2
            x = A * np.sin(a * tau + delta)
            y = B * np.sin(b * tau)
            z = self.z_fixed + C * np.sin(c * tau)
            return [x, y, z]

        # --- Pattern 13: High-frequency zig-zag ---
        elif p == 13:
            Xmax, Ymax = 3.0, 3.0
            freq = 2.0  # zigzag switching frequency (Hz)
            x = Xmax if np.sin(2*np.pi*freq*t) > 0 else -Xmax
            y = Ymax if np.sin(2*np.pi*freq*t + np.pi/2) > 0 else -Ymax
            z = self.z_fixed
            return [x, y, z]

        # --- Pattern 14: Random continuous jitter ---
        elif p == 14:
            mag = 1.0
            if not hasattr(self, '_jitter_pos'):
                self._jitter_pos = np.array([0.0,0.0, self.z_fixed])
            step = mag * np.random.randn(3) * (1.0/self.sim_time)
            self._jitter_pos += step
            self._jitter_pos[0] = np.clip(self._jitter_pos[0], -3, 3)
            self._jitter_pos[1] = np.clip(self._jitter_pos[1], -3, 3)
            self._jitter_pos[2] = np.clip(self._jitter_pos[2], 0.5, 2.5)
            return self._jitter_pos.tolist()

        # --- Pattern 15: Aggressive swoop (dive and climb) ---
        elif p == 15:
            T = self.sim_time
            if t < T/2:
                z = 2.5 - 4.0*(t/(T/2))
            else:
                z = 0.5 + 4.0*((t-T/2)/(T/2))
            R = 2.0
            w = 1.5
            x = R * np.cos(w * t)
            y = R * np.sin(w * t)
            return [x, y, z]

        # --- Pattern 16: Single step to [1,1,1] and hold ---
        elif p == 16:
            # Step at t==0: return origin, thereafter hold at [1,1,1]
            if t <= 0.0:
                return [0.0, 0.0, 0.0]
            else:
                return [1.0, 1.0, 2.0]

        # default: hover at origin
        else:
            return [0.0, 0.0, self.z_fixed]
