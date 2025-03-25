import numpy as np
from control import lqr  # Requires the python-control library
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random

# Connect to the server
client = RemoteAPIClient() 
sim = client.require('sim')

#Handles
propellerHandle = np.array([None, None, None, None])   
jointHandle = np.array([None, None, None, None])
forceSensor = np.array([None, None, None, None])
droneHandle = sim.getObject('/Quadcopter')
targetHandle = sim.getObject('/target')
for i in range(4):
    propellerHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
    jointHandle[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')
    forceSensor[i] = sim.getObject(f'/Quadcopter/propeller[{i}]')






    

sim.setStepping(True)
sim.startSimulation()
while (t := sim.getSimulationTime()) < 500:
    print(f'Simulation time: {t:.2f} [s]')






    sim.step()
sim.stopSimulation()