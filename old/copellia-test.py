from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import random

# Connect to the server
client = RemoteAPIClient() 
sim = client.require('sim')


particlesAreVisible = True
simulateParticles = True
fakeShadow = True

particleCountPerSecond = 430
particleSize = 0.005
particleDensity = 8500
particleScatteringAngle = 30
particleLifeTime = 0.5
maxParticleCount = 50

# Global object handles and control variables (will be initialized in sysCall_init):
targetObj = None
d = None
propellerHandles = [None] * 4
jointHandles = [None] * 4
particleObjects = [-1, -1, -1, -1]
heli = None

pParam = 2
iParam = 0
dParam = 0
vParam = -2

cumul = 0
lastE = 0
pAlphaE = 0
pBetaE = 0
psp2 = 0
psp1 = 0

prevEuler = 0

shadowCont = None

def handlePropeller(index, particleVelocity):
    """
    Handles the operation of a single propeller:
      - Updates the joint position.
      - Creates particle items if simulation of particles is enabled.
      - Applies reactive force and torque to simulate particle effects.
    """
    global particleDensity, particleSize, particleScatteringAngle, simulateParticles, particleCountPerSecond

    propellerRespondable = propellerHandles[index]
    propellerJoint = jointHandles[index]
    # Get the parent object of the respondable part:
    propeller = sim.getObjectParent(propellerRespondable)
    particleObject = particleObjects[index]

    maxParticleDeviation = math.tan(particleScatteringAngle * 0.5 * math.pi / 180) * particleVelocity
    notFullParticles = 0  # This variable is used to accumulate the fractional part

    t = sim.getSimulationTime()
    sim.setJointPosition(propellerJoint, t * 10)
    ts = sim.getSimulationTimeStep()

    m = sim.getObjectMatrix(propeller)
    particleCnt = 0
    pos = [0, 0, 0]
    dir = [0, 0, 1]

    # Determine how many particles should be generated this time step:
    requiredParticleCnt = particleCountPerSecond * ts + notFullParticles
    notFullParticles = requiredParticleCnt - math.floor(requiredParticleCnt)
    requiredParticleCnt = math.floor(requiredParticleCnt)

    # Generate particles in a uniformly random distribution within a unit circle:
    while particleCnt < requiredParticleCnt:
        x = (random.random() - 0.5) * 2
        y = (random.random() - 0.5) * 2
        if (x * x + y * y) <= 1:
            if simulateParticles:
                pos[0] = x * 0.08
                pos[1] = y * 0.08
                pos[2] = -particleSize * 0.6
                dir[0] = pos[0] + (random.random() - 0.5) * maxParticleDeviation * 2
                dir[1] = pos[1] + (random.random() - 0.5) * maxParticleDeviation * 2
                dir[2] = pos[2] - particleVelocity * (1 + 0.2 * (random.random() - 0.5))
                # Transform pos and dir with the propeller's matrix:
                posTrans = sim.multiplyVector(m, pos)
                dirTrans = sim.multiplyVector(m, dir)
                itemData = [posTrans[0], posTrans[1], posTrans[2],
                            dirTrans[0], dirTrans[1], dirTrans[2]]
                sim.addParticleObjectItem(particleObject, itemData)
            particleCnt += 1

    # Calculate and apply the reactive force:
    totalExertedForce = (particleCnt * particleDensity * particleVelocity *
                         math.pi * (particleSize ** 3) / (6 * ts))
    force = [0, 0, totalExertedForce]

    # We remove the translational part from the matrix before applying force/torque:
    m_copy = m[:]  # shallow copy of matrix list
    # In Lua m[4], m[8], m[12] (1-indexed) are set to 0.
    m_copy[3] = 0   # index 3 in Python corresponds to m[4] in Lua
    m_copy[7] = 0   # index 7 corresponds to m[8]
    m_copy[11] = 0  # index 11 corresponds to m[12]
    force = sim.multiplyVector(m_copy, force)

    # Determine rotation direction: in Lua: 1 - math.mod(index,2)*2
    rotDir = 1 - (index % 2) * 2
    torque = [0, 0, rotDir * 0.002 * particleVelocity]
    torque = sim.multiplyVector(m_copy, torque)

    sim.addForceAndTorque(propellerRespondable, force, torque)
def sysCall_init():
    """Initialization function. Called once at the start of the simulation."""
    global targetObj, d, propellerHandles, jointHandles, particleObjects, heli
    global pParam, iParam, dParam, vParam, cumul, lastE, pAlphaE, pBetaE, psp2, psp1, prevEuler, shadowCont

    # Detach the manipulation sphere:
    targetObj = sim.getObject('/target')
    sim.setObjectParent(targetObj, -1, True)

    # Get the base object:
    d = sim.getObject('/Quadcopter/base')

    # Create particle objects:
    # Prepare the particle type by combining flags:
    ttype = (sim.particle_roughspheres +
             sim.particle_cyclic +
             sim.particle_respondable1to4 +
             sim.particle_respondable5to8 +
             sim.particle_ignoresgravity)
    if not particlesAreVisible:
        ttype += sim.particle_invisible

    # Loop over 4 propellers (Lua indices 1–4 become Python 0–3):

    for i in range(4):
        propellerHandles[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/respondable')
        jointHandles[i] = sim.getObject(f'/Quadcopter/propeller[{i}]/joint')
        if simulateParticles:
            particleObjects[i] = sim.addParticleObject(
                ttype,
                particleSize,
                particleDensity,
                [2, 1, 0.2, 3, 0.4],
                particleLifeTime,
                maxParticleCount,
                [0.3, 0.7, 1]
            )

    heli = sim.getObject('/Quadcopter')

    # Controller parameters (already set above but could be reinitialized here if needed)
    pParam = 2
    iParam = 0
    dParam = 0
    vParam = -2

    cumul = 0
    lastE = 0
    pAlphaE = 0
    pBetaE = 0
    psp2 = 0
    psp1 = 0
    prevEuler = 0

    # Create a drawing object for a fake shadow if enabled:
    if fakeShadow:
        shadowCont = sim.addDrawingObject(
            sim.drawing_discpts +
            sim.drawing_cyclic +
            sim.drawing_25percenttransparency +
            sim.drawing_50percenttransparency +
            sim.drawing_itemsizes,
            0.2, 0, -1, 1
        )
    

sim.setStepping(True)
sim.startSimulation()
sysCall_init()
while (t := sim.getSimulationTime()) < 500:
    print(f'Simulation time: {t:.2f} [s]')

    # Get the position of the base object:
    pos = sim.getObjectPosition(d)
    if fakeShadow:
        # Draw a fake shadow:
        # The itemData array contains: x, y, z, and additional parameters
        itemData = [pos[0], pos[1], 0.002, 0, 0, 0, 1, 0.2]
        sim.addDrawingObjectItem(shadowCont, itemData)

    # -- Vertical control --
    targetPos = sim.getObjectPosition(targetObj)
    pos = sim.getObjectPosition(d)
    l = sim.getVelocity(heli)
    # In Lua, the z-component is the 3rd element; in Python it is at index 2.
    e = targetPos[2] - pos[2]
    cumul += e
    pv = pParam * e
    thrust = 5.45 + pv + iParam * cumul + dParam * (e - lastE) + l[0][2] * vParam
    lastE = e

    # -- Horizontal control --
    # sp is the target position expressed in the local frame of d:
    sp = sim.getObjectPosition(targetObj, d)
    m = sim.getObjectMatrix(d)
    # In Lua, vx = {1,0,0} and vy = {0,1,0}; then they are transformed:
    vx = sim.multiplyVector(m, [1, 0, 0])
    vy = sim.multiplyVector(m, [0, 1, 0])
    # In Lua, m[12] is the third translation element; in Python, assuming a flat list of 12 elements, it is at index 11.
    alphaE = vy[2] - m[11]
    alphaCorr = 0.25 * alphaE + 2.1 * (alphaE - pAlphaE)
    betaE = vx[2] - m[11]
    betaCorr = -0.25 * betaE - 2.1 * (betaE - pBetaE)
    pAlphaE = alphaE
    pBetaE = betaE
    # Adjust using the target’s x and y components (note Lua indices: sp[2] is y, sp[1] is x):
    alphaCorr += sp[1] * 0.005 + 1 * (sp[1] - psp2)
    betaCorr  -= sp[0] * 0.005 + 1 * (sp[0] - psp1)
    psp2 = sp[1]
    psp1 = sp[0]

    # -- Rotational control --
    # Get Euler angles of d relative to targetObj. Lua’s euler[3] becomes euler[2] in Python (0-indexed)
    euler = sim.getObjectOrientation(d, targetObj)
    rotCorr = euler[2] * 0.1 + 2 * (euler[2] - prevEuler)
    prevEuler = euler[2]

    # -- Decide motor velocities --
    # Note: In Lua, handlePropeller is called with indices 1,2,3,4. In Python we use 0–3.
    handlePropeller(0, thrust * (1 - alphaCorr + betaCorr + rotCorr))
    handlePropeller(1, thrust * (1 - alphaCorr - betaCorr - rotCorr))
    handlePropeller(2, thrust * (1 + alphaCorr - betaCorr + rotCorr))
    handlePropeller(3, thrust * (1 + alphaCorr + betaCorr - rotCorr))







    sim.step()
sim.stopSimulation()