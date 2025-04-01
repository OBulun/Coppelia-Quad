# noise.py
import numpy as np

def add_modeling_noise(state_dot, noise_std):
    """
    Add modeling noise to the state derivative.
    
    Parameters:
      state_dot : numpy array
          The original state derivative vector.
      noise_std : float or numpy array
          Standard deviation(s) for the modeling noise.
    
    Returns:
      numpy array
          The state derivative with added modeling noise.
    """
    noise = np.random.normal(0, noise_std, size=state_dot.shape)
    return state_dot + noise

def add_measurement_noise(state, noise_std):
    """
    Add measurement noise to the state vector.
    
    Parameters:
      state : numpy array
          The true state vector.
      noise_std : float or numpy array
          Standard deviation(s) for the measurement noise.
    
    Returns:
      numpy array
          The state vector with added measurement noise.
    """
    noise = np.random.normal(0, noise_std, size=state.shape)
    return state + noise

def apply_disturbance(drone, disturbance_params):
    """
    Apply disturbances to the drone model.
    This function modifies the drone's parameters in place.
    
    For example, you can add extra mass to model a battery weight or
    add a constant extra force to simulate wind gusts.
    
    Parameters:
      drone : instance of Drone
          An instance of the Drone class (from drone.py).
      disturbance_params : dict
          A dictionary of disturbance parameters. Examples:
              {"extra_mass": 0.2}  
                -> Increase the drone's mass by 0.2 kg (e.g. battery weight)
              {"extra_force": np.array([0, 0, -0.5])}
                -> Apply an extra constant force in the inertial frame.
                
    Returns:
      None
    """
    # Apply extra mass disturbance (e.g., battery weight)
    if "extra_mass" in disturbance_params:
        drone.m += disturbance_params["extra_mass"]
    
    # Apply extra force disturbance
    # Note: To use this extra force in the dynamics, you'll need to modify your
    # Drone.dynamics method to include drone.extra_force.
    if "extra_force" in disturbance_params:
        drone.extra_force = disturbance_params["extra_force"]
    else:
        # Ensure the attribute exists (defaults to zero force)
        drone.extra_force = np.zeros(3)
