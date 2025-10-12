# Coppelia-Quad

A quadcopter simulation and control project using Model Predictive Control (MPC) and PID controllers in CoppeliaSim. This project implements various trajectory tracking algorithms for autonomous quadcopter flight with disturbance rejection and sensor noise simulation.

## 🎯 Overview

This project implements a sophisticated control system for quadcopter trajectory tracking in CoppeliaSim. It combines:

- **Linear MPC (Model Predictive Control)** for horizontal position and attitude control
- **PID Controller** for altitude (Z-axis) control
- **Disturbance estimation and rejection** using integral action
- **Sensor noise simulation** for realistic testing
- **Multiple trajectory patterns** including circles, helixes, Lissajous curves, and custom waypoints

### Key Capabilities

- Real-time trajectory tracking with sub-meter accuracy
- Wind disturbance rejection (configurable X/Y components)
- Sensor noise simulation with configurable standard deviation
- Multiple pre-defined flight patterns (20+ patterns)
- Comprehensive data logging for analysis
- Visualization tools for trajectory plotting

## ✨ Features

### Control Features
- **Linear MPC Controller**: 
  - 12-state system (position, velocity, attitude, angular rates)
  - 4 control inputs (thrust deviation, roll/pitch/yaw torques)
  - Configurable prediction horizon (default: 10 steps)
  - Constraint handling for actuator limits
  - Warm-start optimization with OSQP solver

- **PID Altitude Control**:
  - Independent Z-axis control with anti-windup
  - Tunable gains (Kp, Ki, Kd)
  - Integrated with MPC for complete 3D control

- **Disturbance Rejection**:
  - Integral action on position error
  - Wind force simulation (X/Y components)
  - External disturbance estimation

### Simulation Features
- **Sensor Noise Simulation**: Toggle-able Gaussian noise on position measurements
- **Wind Simulation**: Configurable wind forces in X/Y directions (experimental)
- **Real-time Logging**: CSV export of states, controls, forces, and targets
- **Visualization**: HTML reports and plotting scripts

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CoppeliaSim Environment                  │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Quadcopter  │◄────────│    Target    │                  │
│  │   (Drone)    │         │  (Pattern)   │                  │
│  └──────┬───────┘         └──────────────┘                  │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │ ZMQ Remote API
          │
┌─────────▼────────────────────────────────────────────────────┐
│                    Python Control System                     │
│                                                              │
│  ┌────────────────┐      ┌─────────────────┐                │
│  │  State         │─────►│  MPC Controller │                │
│  │  Estimation    │      │  (X/Y Position, │                │
│  │  + Noise       │      │   Attitude)     │                │
│  └────────────────┘      └────────┬────────┘                │
│                                   │                          │
│  ┌────────────────┐               │                          │
│  │  PID           │◄──────────────┤                          │
│  │  Controller    │               │                          │
│  │  (Z Altitude)  │               │                          │
│  └────────┬───────┘               │                          │
│           │                       │                          │
│           ▼                       ▼                          │
│  ┌──────────────────────────────────┐                        │
│  │   Force/Torque Allocation        │                        │
│  │   (4 Propellers)                 │                        │
│  └──────────────┬───────────────────┘                        │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────────────┐                        │
│  │   Logging & Visualization        │                        │
│  └──────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

## � Quick Start

**Get up and running in 5 minutes:**

```bash
# 1. Clone the repository
git clone https://github.com/OBulun/Coppelia-Quad.git
cd Coppelia-Quad

# 2. Install dependencies
pip install numpy cvxpy matplotlib pandas coppeliasim-zmqremoteapi-client

# 3. Open CoppeliaSim and load drone_scene.ttt

# 4. Run the simulation
python MPC_v4.0.0.py

# 5. View results
python scripts/plot_trajectory_xyz.py
```

**Expected Output:**
- Console logs showing simulation progress
- `simulation_logs.csv` with timestamped data
- Plots showing trajectory tracking performance

##  Installation

### Prerequisites

1. **CoppeliaSim** (formerly V-REP)
   - Download from [CoppeliaSim website](https://www.coppeliarobotics.com/)
   - Ensure ZMQ Remote API is enabled

2. **Python 3.8+** 

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OBulun/Coppelia-Quad.git
   cd Coppelia-Quad
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n quad_sim python=3.9
   conda activate quad_sim
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy cvxpy matplotlib pandas
   pip install coppeliasim-zmqremoteapi-client
   ```

4. **Open the CoppeliaSim scene**:
   - Launch CoppeliaSim
   - Open `drone_scene.ttt`

## 🚀 Usage

### Basic Simulation

1. **Start CoppeliaSim** and load `drone_scene.ttt`

2. **Run the main controller**:
   ```bash
   python MPC_v4.0.0.py
   ```

3. **View results**:
   - Logs are saved to `simulation_logs.csv`
   - Parameters saved to `simulation_parameters.txt`
   - Latest HTML report in `logs/simulation_logs_latest.html`

### Running Different Versions

The project includes multiple MPC implementations:

```bash
# Linear MPC with PID altitude control (Recommended)
python MPC_v4.0.0.py

# Latest experimental version
python MPC_v5.py

# Other versions with different features
python MPC_v4.6.py
```

### Plotting Results

```bash
# Plot XY trajectory
python scripts/plot_trajectory_xy.py

# Plot 3D trajectory
python scripts/plot_trajectory_xyz.py

# Full plotting suite
python scripts/plotting.py
```

## 📁 Project Structure

```
Coppelia-Quad/
├── MPC_v4.0.0.py              # Main controller (Linear MPC + PID)
├── drone_scene.ttt            # CoppeliaSim scene file
│
├── functions/                 # Core library modules
│   ├── __init__.py
│   ├── MPC_controller.py      # Basic MPC implementation
│   ├── mpc_dist.py            # MPC with disturbance rejection
│   ├── MPC_exp.py             # Experimental MPC
│   ├── PID_Controller.py      # PID controller class
│   ├── patterns.py            # Trajectory pattern generator
│   ├── noise.py               # Sensor noise utilities
│   ├── save_logs.py           # Data logging functions
│   └── save_parameters.py     # Parameter export
│
├── scripts/                   # Analysis and visualization
│   ├── plot_trajectory_xy.py  # 2D trajectory plots
│   ├── plot_trajectory_xyz.py # 3D trajectory plots
│   └── plotting.py            # Comprehensive plotting suite
│
├── .gitignore
└── README.md
```

## 🎮 Control Algorithms

### Model Predictive Control (MPC)

The MPC controller uses a **linearized 12-state model**:

**State Vector** (x ∈ ℝ¹²):
```
x = [x, y, z, vₓ, vᵧ, vᵤ, φ, θ, ψ, p, q, r]ᵀ
```
- Position: `(x, y, z)` [m]
- Linear velocity: `(vₓ, vᵧ, vᵤ)` [m/s]
- Attitude: `(φ, θ, ψ)` (roll, pitch, yaw) [rad]
- Angular rates: `(p, q, r)` [rad/s]

**Control Input** (u ∈ ℝ⁴):
```
u = [Δf, τᵩ, τθ, τᵨ]ᵀ
```
- `Δf`: Thrust deviation from hover [N]
- `τᵩ, τθ, τᵨ`: Roll, pitch, yaw torques [N·m]

**Optimization Problem**:
```
minimize   Σ(k=0 to N-1) [(xₖ - xᵣₑf)ᵀQ(xₖ - xᵣₑf) + uₖᵀRuₖ] + (xₙ - xᵣₑf)ᵀQ(xₙ - xᵣₑf)
subject to xₖ₊₁ = Aₓₖ + Buₖ
           uₘᵢₙ ≤ uₖ ≤ uₘₐₓ
           x₀ = x(t)  (current state)
```

**Solver**: OSQP (Operator Splitting Quadratic Program) with warm-start

### PID Altitude Control

Independent PID controller for Z-axis:
```
u_z(t) = Kₚe(t) + Kᵢ∫e(τ)dτ + Kₐde(t)/dt
```
where `e(t) = z_target - z_current`

Features:
- Anti-windup with configurable guard limits
- Integrated into MPC framework
- Tuned for fast altitude response

### Disturbance Estimation

Position error integral for external disturbance rejection:
```
d̂ₖ₊₁ = d̂ₖ + λ(xₜₐᵣgₑₜ - xₖ)Δt
```
where `λ` is the integrator gain (default: 0.3)

## 🛤️ Trajectory Patterns

The `PatternGenerator` class supports 20+ trajectory patterns:

| Pattern ID | Description | Characteristics |
|------------|-------------|-----------------|
| 0 | Christmas Tree Spiral | Rising spiral with end hold |
| 1 | Rectangular Path | Square trajectory at fixed altitude |
| 2 | Circle (XY plane) | Circular motion with sine wave Z |
| 3 | Helix | 3D spiral upward |
| 4 | Figure-8 (Lemniscate) | Horizontal figure-8 pattern |
| 5 | Vertical Sine Wave | Oscillating altitude |
| 6 | Diagonal Line | Linear trajectory |
| 7 | Random Waypoints | Random points with interpolation |
| 8 | Trefoil Knot | 3D mathematical knot |
| 9 | Square Spiral | Expanding square pattern |
| 10 | Star Pattern | 5-pointed star |
| 11 | Infinity (3D) | Vertical figure-8 |
| 12 | Lissajous Curve | Parametric harmonic motion |
| 13 | Rose Curve | Mathematical rose pattern |
| 14 | Hypocycloid | Rolling circle curve |
| 15 | Epitrochoid | Epicycle pattern |
| 16 | Wave Path | Sinusoidal horizontal wave |
| 17 | Zigzag | Sharp directional changes |
| 18 | Circular Ascent | Rising circular motion |
| 19 | Cloverleaf | 4-leaf clover pattern |
| 20 | Double Helix | DNA-like double spiral |

## 📚 Version History

### MPC_v4.0.0.py *(Recommended)*
- **Features**: Linear MPC + PID altitude control
- **Stability**: High
- **Performance**: Excellent tracking for moderate speeds
- **Sensor Noise**: Toggle-able simulation
- **Status**: ✅ Production ready

## � Future Work

### IDEA-1: Dynamic Window Approach (DWA)
- Implement DWA as a high-level path planner
- Use DWA to find safe and smooth state references between targets
- Use MPC as low-level controller to track DWA references
- Enable obstacle avoidance and dynamic re-planning

### Other Improvements
- [ ] Nonlinear MPC for improved accuracy at high speeds
- [ ] Adaptive MPC with online parameter estimation
- [ ] Multi-drone coordination and formation control
- [ ] Vision-based state estimation
- [ ] Real hardware deployment (PX4/ArduPilot integration)
- [ ] Reinforcement learning for automatic tuning
- [ ] ROS2 integration for modularity
- [ ] GPU acceleration for real-time nonlinear MPC
- [ ] Extended Kalman Filter (EKF) for state estimation
- [ ] Collision avoidance with multiple obstacles

## 📦 Dependencies

### Core Dependencies
```
numpy >= 1.20.0          # Numerical computing
cvxpy >= 1.2.0           # Convex optimization
matplotlib >= 3.3.0      # Plotting and visualization
pandas >= 1.2.0          # Data manipulation
```

### CoppeliaSim API
```
coppeliasim-zmqremoteapi-client >= 1.0.0
```

### Solvers
- **OSQP** (installed with cvxpy): Quadratic program solver


### Installation Command
```bash
pip install numpy cvxpy matplotlib pandas coppeliasim-zmqremoteapi-client
```

## 📊 Data Logging

The simulation logs the following data:

- **Time Log**: Simulation timestamps
- **State Log**: Full 12-state vectors
- **Control Log**: 4D control inputs (thrust, torques)
- **Force Log**: Individual propeller forces [f₀, f₁, f₂, f₃]
- **Target Log**: Reference positions [x, y, z]
- **Distance Log**: Euclidean distance to target

Logs are saved in CSV format and can be visualized using the provided scripts.

### Visualization Examples

```bash
# Generate all plots (XY, XYZ, state history, control inputs)
python scripts/plotting.py

# Quick 2D trajectory comparison
python scripts/plot_trajectory_xy.py

# 3D interactive plot
python scripts/plot_trajectory_xyz.py
```

**Generated Plots Include:**
- Position tracking (actual vs. reference)
- Velocity profiles
- Attitude angles (roll, pitch, yaw)
- Control inputs over time
- Tracking error statistics
- Force allocation to propellers
















