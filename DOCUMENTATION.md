# Technical Documentation

## Table of Contents
- [System Architecture](#system-architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Reinforcement Learning Setup](#reinforcement-learning-setup)
- [Vehicle Dynamics](#vehicle-dynamics)
- [Sensor Models](#sensor-models)
- [Control System](#control-system)
- [Training Process](#training-process)
- [Code Structure](#code-structure)
- [API Reference](#api-reference)

---

## System Architecture

### Overview

The system follows a modular architecture with clear separation between:
1. **RL Environment** - Interfaces with Simulink model
2. **Vehicle Dynamics** - Physics simulation
3. **Sensor Suite** - Measurement simulation
4. **Control System** - Flight controller
5. **Visualization** - Real-time monitoring

```
┌─────────────────────────────────────────────────────────┐
│                   MATLAB RL Agent                       │
│                    (PPO Algorithm)                      │
└──────────────────┬──────────────────────────────────────┘
                   │ Actions (thrust commands)
                   │ Observations (state)
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Simulink Environment                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Flight Control System                          │   │
│  │  - Altitude Controller (PPO/Fuzzy/PID)         │   │
│  │  - Attitude Controller                          │   │
│  │  - Control Mixer                                │   │
│  └────────────┬────────────────────────────────────┘   │
│               │ Motor Commands                          │
│               ▼                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Vehicle Dynamics (6-DOF)                       │   │
│  │  - Nonlinear Airframe Model                     │   │
│  │  - Rotor Aerodynamics                           │   │
│  │  - Motor Dynamics                               │   │
│  └────────────┬────────────────────────────────────┘   │
│               │ True State                              │
│               ▼                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Sensor Suite                                   │   │
│  │  - IMU (Accel + Gyro)                          │   │
│  │  - Sonar Altimeter                             │   │
│  │  - Optical Flow                                │   │
│  └────────────┬────────────────────────────────────┘   │
│               │ Measurements                            │
│               ▼                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  State Estimation                               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Hierarchy

**Top Level**: `parrotMinidroneHover.slx`
- Command Input
- Flight Control System (`controller/flightControlSystem.slx`)
- Airframe (`linearAirframe/` or `nonlinearAirframe/`)
- Sensors
- Visualization

---

## Mathematical Foundations

### Coordinate Systems

#### Body Frame (B)
- **Origin**: Center of mass
- **X-axis**: Forward (nose direction)
- **Y-axis**: Right wing
- **Z-axis**: Downward (gravity direction)

#### Inertial Frame (I)
- **Origin**: Ground reference point
- **X-axis**: North
- **Y-axis**: East
- **Z-axis**: Down (NED convention)

### State Vector

The complete state vector has 12 components:

```
x = [p_x, p_y, p_z, u, v, w, φ, θ, ψ, p, q, r]ᵀ
```

Where:
- **Position** (I-frame): `p_x, p_y, p_z` [m]
- **Velocity** (B-frame): `u, v, w` [m/s]
- **Attitude** (Euler): `φ (roll), θ (pitch), ψ (yaw)` [rad]
- **Angular velocity** (B-frame): `p, q, r` [rad/s]

### Equations of Motion

#### Translational Dynamics
```
m * dv/dt = F_aero + F_thrust + F_gravity + F_ext
```

#### Rotational Dynamics
```
I * dω/dt + ω × (I * ω) = M_aero + M_thrust + M_ext
```

Where:
- `m` = mass [kg]
- `I` = inertia tensor [kg⋅m²]
- `v` = velocity vector [m/s]
- `ω` = angular velocity vector [rad/s]
- `F` = force vectors [N]
- `M` = moment vectors [N⋅m]

### Aerodynamics Model

#### Rotor Thrust
```
T_i = k_T * ω_i²
```

Where:
- `T_i` = thrust of rotor i [N]
- `k_T` = thrust coefficient
- `ω_i` = rotor angular velocity [rad/s]

#### Drag Force
```
F_drag = -0.5 * ρ * A * C_d * v * |v|
```

Where:
- `ρ` = air density [kg/m³]
- `A` = reference area [m²]
- `C_d` = drag coefficient
- `v` = velocity vector [m/s]

---

## Reinforcement Learning Setup

### Environment Definition

**Type**: Custom Simulink Environment
**File**: `parrotPPOenv.mlx`

#### Observation Space

The agent observes:
```matlab
Observations = [
    z_error,      % Altitude error [m]
    z_dot,        % Vertical velocity [m/s]
    theta,        % Pitch angle [rad]
    phi,          % Roll angle [rad]
    theta_dot,    % Pitch rate [rad/s]
    phi_dot       % Roll rate [rad/s]
]
```

**Dimension**: 6
**Type**: Continuous
**Range**: Normalized to [-1, 1]

#### Action Space

The agent outputs:
```matlab
Action = thrust_command  % Collective thrust adjustment
```

**Dimension**: 1
**Type**: Continuous
**Range**: [-1, 1] (mapped to motor command range)

#### Reward Function

**Implementation**:
```matlab
% Altitude error component
e = abs(z - z_desired);
R_altitude = -w1 * exp(-(1/(alpha * e))^2);

% Vertical velocity component
R_velocity = -w2 * exp(-(1/(beta * z_dot))^2);

% Total reward
R = R_altitude + R_velocity;
```

**Parameters**:
- `w1 = 1.0` - Altitude error weight
- `w2 = 0.1` - Velocity penalty weight
- `alpha = 0.5` - Altitude error scaling
- `beta = 1.0` - Velocity scaling

**Rationale**:
- Exponential decay provides smooth gradients
- Penalizes large errors more heavily
- Encourages smooth approach to target
- Prevents oscillations via velocity penalty

### PPO Algorithm Configuration

#### Network Architecture

**Actor Network**:
```
Input (6) → FC(128, tanh) → FC(128, tanh) → Output(1, tanh)
```

**Critic Network**:
```
Input (6) → FC(128, tanh) → FC(128, tanh) → Output(1, linear)
```

#### Training Hyperparameters

```matlab
% PPO Options
MaxEpisodes = 1000
MaxStepsPerEpisode = 2000  % 100s @ 5ms timestep
ScoreAveragingWindowLength = 100

% Learning rates
ActorLearnRate = 1e-4
CriticLearnRate = 5e-4

% PPO specific
ClipFactor = 0.2
EntropyLossWeight = 0.01
MiniBatchSize = 128
NumEpoch = 3

% Experience buffer
ExperienceHorizon = 2048
DiscountFactor = 0.99
GAEFactor = 0.95

% GPU acceleration
UseDevice = 'gpu'
```

#### Training Termination Conditions

1. **Success**: Average reward > -0.1 over 100 episodes
2. **Failure**: Altitude error > 50m
3. **Timeout**: Max episodes reached
4. **Crash**: Any position NaN or Inf

---

## Vehicle Dynamics

### Parrot Mambo Parameters

```matlab
% Mass properties
mass = 0.063;  % kg
inertia = [
    5.8286e-5,  0,           0;
    0,           7.1691e-5,  0;
    0,           0,           1.0e-4
];  % kg⋅m²

% Geometry
arm_length = 0.055;  % m (motor to center)
motor_diameter = 0.046;  % m

% Motor parameters
motor_time_constant = 0.02;  % s
motor_command_min = 10;
motor_command_max = 500;

% Aerodynamics
k_thrust = 1.0e-6;  % Thrust coefficient
k_drag = 2.5e-9;    % Drag coefficient
```

### Parrot Rolling Spider Parameters

```matlab
% Mass properties
mass = 0.068;  % kg
inertia = [
    6.8600e-5,  0,           0;
    0,           9.2000e-5,  0;
    0,           0,           1.3660e-4
];  % kg⋅m²

% Similar geometry to Mambo
```

### Motor Layout

Quadcopter X-configuration:
```
     Front
       ↑
   M1     M2
     \ ⊕ /
      \ /
      / \
     / ⊕ \
   M4     M3
```

**Rotation directions**:
- M1 (front-left): CCW
- M2 (front-right): CW
- M3 (rear-right): CCW
- M4 (rear-left): CW

### Control Mixer

Maps control commands to motor thrusts:

```matlab
% Control vector: [Thrust, Roll, Pitch, Yaw]
u = [T; τ_φ; τ_θ; τ_ψ];

% Mixer matrix
M = [
    1,  -1/√2,  -1/√2,  -1;  % Motor 1
    1,   1/√2,  -1/√2,   1;  % Motor 2
    1,   1/√2,   1/√2,  -1;  % Motor 3
    1,  -1/√2,   1/√2,   1;  % Motor 4
];

% Motor commands
motor_cmd = M * u;
```

---

## Sensor Models

### IMU (Inertial Measurement Unit)

#### Accelerometer

**Model**:
```matlab
a_measured = a_true + bias + noise
```

**Parameters**:
- **Range**: ±156.96 m/s² (±16g)
- **Resolution**: 0.0048 m/s² (0.488 mg)
- **Noise density**: 0.0003924 m/s²/√Hz
- **Bias instability**: 0.049 m/s² (5 mg)
- **Temperature coefficient**: ±0.005%/°C
- **Sampling rate**: 200 Hz

#### Gyroscope

**Model**:
```matlab
ω_measured = ω_true + bias + noise + scale_factor_error
```

**Parameters**:
- **Range**: ±8.7266 rad/s (±500°/s)
- **Resolution**: 0.00013 rad/s (0.00762°/s)
- **Noise density**: 0.0001 rad/s/√Hz
- **Bias instability**: 0.0035 rad/s (0.2°/s)
- **Scale factor**: 0.3% typical
- **Sampling rate**: 200 Hz

### Sonar Altimeter

**Model**:
```matlab
h_measured = h_true + noise (if h < h_max)
```

**Parameters**:
- **Range**: 0.2 to 3.0 m
- **Resolution**: 0.01 m (1 cm)
- **Accuracy**: ±0.05 m
- **Update rate**: 20 Hz
- **Beam angle**: 25° cone
- **Noise**: Gaussian, σ = 0.02 m

**Limitations**:
- Returns NaN if altitude > 3m
- Affected by surface reflectivity
- Multipath errors near walls

### Optical Flow Sensor

**Model**:
```matlab
flow = [v_x; v_y] / h + noise
```

**Parameters**:
- **Resolution**: 0.1 rad/s
- **Range**: 0.2 to 3.0 m altitude
- **Frame rate**: 200 Hz
- **FOV**: 42° diagonal
- **Noise**: σ = 0.05 rad/s

---

## Control System

### Architecture

The flight control system uses a cascaded structure:

```
Target       Outer Loop         Inner Loop        Mixer        Motors
Altitude  →  Altitude Ctrl  →  Attitude Ctrl  →  Control  →  Motor
              (PPO/Fuzzy)       (PID)             Mixer       Dynamics
```

### Altitude Controller (RL Agent)

**Input**:
- Altitude error
- Vertical velocity
- Attitude (pitch, roll)
- Attitude rates

**Output**:
- Thrust command adjustment

**Update rate**: 200 Hz (5ms)

### Attitude Controller (PID)

**Roll/Pitch Controller**:
```matlab
% P-controller for angles
τ_φ = K_p_φ * (φ_desired - φ)
τ_θ = K_p_θ * (θ_desired - θ)

% PD-controller for rates
τ_p = K_p_p * (p_desired - p) + K_d_p * dp/dt
τ_q = K_p_q * (q_desired - q) + K_d_q * dq/dt
```

**Yaw Controller**:
```matlab
τ_ψ = K_p_ψ * (ψ_desired - ψ) + K_d_ψ * r
```

**Gains** (Mambo):
```matlab
K_p_φ = 6.0
K_p_θ = 6.0
K_p_ψ = 3.0

K_p_p = 0.4
K_d_p = 0.05

K_p_q = 0.4
K_d_q = 0.05

K_d_ψ = 0.3
```

---

## Training Process

### Training Pipeline

1. **Environment Setup**
   ```matlab
   % Initialize Simulink model
   load_system('parrotMinidroneHover');

   % Create RL environment
   env = createEnvironment();
   ```

2. **Agent Creation**
   ```matlab
   % Define observation and action specs
   obsInfo = getObservationInfo(env);
   actInfo = getActionInfo(env);

   % Create actor-critic networks
   actor = createActor(obsInfo, actInfo);
   critic = createCritic(obsInfo);

   % Create PPO agent
   agent = rlPPOAgent(actor, critic);
   ```

3. **Training Options**
   ```matlab
   trainOpts = rlTrainingOptions(...
       'MaxEpisodes', 1000, ...
       'MaxStepsPerEpisode', 2000, ...
       'Verbose', true, ...
       'Plots', 'training-progress', ...
       'StopTrainingCriteria', 'AverageReward', ...
       'StopTrainingValue', -0.1, ...
       'ScoreAveragingWindowLength', 100, ...
       'SaveAgentCriteria', 'EpisodeReward', ...
       'SaveAgentValue', -0.5);
   ```

4. **Training Execution**
   ```matlab
   % Train the agent
   trainingStats = train(agent, env, trainOpts);

   % Save trained agent
   save('trainedAgent.mat', 'agent');
   ```

### Training History

Based on `Record.txt` and agent files:

#### Session 1: 2022-08-25
- **Reward function**: Exponential decay (squared)
- **Parameters**: α=0.5, β=1.0, w1=1.0, w2=0.1
- **Result**: Saved as `0825.mat`

#### Session 2: 2022-08-29
- **Reward function**: Modified exponential decay
- **Parameters**: Non-squared exponential
- **Early stopping**: e > 50m
- **Result**: Saved as `0829.mat`

#### Session 3: 2022-09-26
- **Adjustments**: Fine-tuned network architecture
- **Result**: Saved as `0926.mat`

#### Session 4: 2022-10-01
- **Final version**: Best performing agent
- **Result**: Saved as `1001.mat`
- **Performance**: Stable altitude control with minimal overshoot

---

## Code Structure

### Configuration Files (tasks/)

#### `vehicleVars.m`
Defines vehicle parameters:
- Mass, inertia
- Geometry
- Aerodynamic coefficients
- Motor characteristics

#### `controllerVars.m`
Defines controller parameters:
- PID gains
- Control limits
- Sampling rates
- Mixer matrix

#### `sensorsVars.m`
Defines sensor specifications:
- Noise parameters
- Bias values
- Update rates
- Range limits

#### `asbBusDefinition*.m`
Defines Simulink bus objects for data exchange:
- `States` - Vehicle state vector
- `Sensors` - Sensor measurements
- `Commands` - Control commands
- `Environment` - Environmental parameters

### Utility Functions (utilities/)

#### `startVars.m`
```matlab
% Initialize all workspace variables
% Set default drone model
% Load configuration files
% Setup paths
```

#### `cleanUpProject.m`
```matlab
% Clear temporary files
% Close models
% Reset workspace
```

#### `generateFlightCode.m`
```matlab
% Configure code generation
% Build embedded code
% Generate deployment package
```

### Model Variants

The Simulink model uses **variant subsystems** for flexibility:

**Command Source**:
- Signal Builder
- Joystick Input
- Pre-recorded Data

**Sensors**:
- Ideal (no dynamics)
- Realistic (with delays and noise)

**Environment**:
- Constant (no wind)
- Variable (with disturbances)

**Visualization**:
- Scopes
- Data logging
- FlightGear
- Simulink 3D Animation

**Vehicle**:
- Linear airframe
- Nonlinear airframe

---

## API Reference

### Key MATLAB Functions

#### Environment Creation
```matlab
env = rlSimulinkEnv(modelName, agentBlock, obsInfo, actInfo)
```

#### Agent Training
```matlab
trainingStats = train(agent, env, trainOpts)
```

#### Agent Simulation
```matlab
simOptions = rlSimulationOptions('MaxSteps', 500);
experience = sim(env, agent, simOptions);
```

#### Agent Evaluation
```matlab
results = evaluateAgent(agent, env, numEpisodes);
```

### Simulink Model Parameters

#### Set Drone Model
```matlab
setMamboModel()           % Configure for Parrot Mambo
setRollingSpiderModel()   % Configure for Rolling Spider
```

#### Run Simulation
```matlab
sim('parrotMinidroneHover', 'StopTime', '100')
```

#### Access Logged Data
```matlab
% After simulation
altitude = logsout.get('altitude').Values;
velocities = logsout.get('velocities').Values;
```

---

## Performance Analysis

### Metrics Calculation

```matlab
% Settling time (2% criterion)
settling_time = calculateSettlingTime(altitude, target, 0.02);

% Overshoot
overshoot = (max(altitude) - target) / target * 100;

% Steady-state error
steady_state_error = abs(mean(altitude(end-100:end)) - target);

% RMS error
rms_error = sqrt(mean((altitude - target).^2));
```

### Typical Performance (1001.mat agent)

| Metric | Value |
|--------|-------|
| Settling Time | 4.2s |
| Overshoot | 8.3% |
| Steady-State Error | 0.07m |
| RMS Error | 0.15m |

---

## Advanced Topics

### Transfer Learning

To adapt the trained agent to new conditions:

```matlab
% Load pre-trained agent
load('1001.mat', 'agent');

% Modify environment parameters
% (e.g., different mass, wind conditions)

% Continue training
newAgent = train(agent, newEnv, trainOpts);
```

### Multi-Agent Training

For formation flight or coordination:

```matlab
% Create multiple environments
env1 = createDroneEnv('Drone1');
env2 = createDroneEnv('Drone2');

% Use parallel experience gathering
% (requires Parallel Computing Toolbox)
```

### Sim-to-Real Transfer

Considerations for hardware deployment:

1. **Domain Randomization**: Vary parameters during training
2. **Robust Sensors**: Add realistic noise models
3. **Safety Constraints**: Implement failsafes
4. **Computational Limits**: Optimize network for embedded device

---

## Troubleshooting Guide

### Common Issues

#### Training Diverges
- **Symptom**: Reward becomes very negative
- **Solutions**:
  - Reduce learning rates
  - Decrease network size
  - Adjust reward function scaling
  - Check observation normalization

#### Agent Oscillates
- **Symptom**: Constant overshooting
- **Solutions**:
  - Increase velocity penalty (w2)
  - Add derivative term to reward
  - Reduce action magnitude
  - Check control saturation

#### Slow Convergence
- **Symptom**: Reward improves very slowly
- **Solutions**:
  - Increase batch size
  - Adjust entropy coefficient
  - Use curriculum learning
  - Verify GPU utilization

#### Simulation Crashes
- **Symptom**: Simulink model stops unexpectedly
- **Solutions**:
  - Check for NaN/Inf values
  - Verify initial conditions
  - Reduce timestep size
  - Enable error logging

---

## References

1. **PPO Algorithm**: Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)

2. **Quadcopter Dynamics**: Beard, R. W., & McLain, T. W. "Small Unmanned Aircraft: Theory and Practice" (2012)

3. **MATLAB Documentation**:
   - Reinforcement Learning Toolbox
   - Aerospace Blockset
   - Simulink

4. **Parrot Minidrone**: Official specifications and documentation

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Author**: Project maintainer
