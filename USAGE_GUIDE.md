# Usage Guide

Complete step-by-step guide for using the Parrot PPO Controller project.

## Table of Contents
- [Getting Started](#getting-started)
- [Training a New Agent](#training-a-new-agent)
- [Testing Pre-trained Agents](#testing-pre-trained-agents)
- [Evaluating Performance](#evaluating-performance)
- [Customizing the Environment](#customizing-the-environment)
- [Hardware Deployment](#hardware-deployment)
- [Visualization Options](#visualization-options)
- [Advanced Usage](#advanced-usage)

---

## Getting Started

### Prerequisites Check

Before starting, verify you have all required software:

```matlab
% Check MATLAB version
version

% Check for required toolboxes
ver

% Check GPU availability (for training)
gpuDevice
```

Required toolboxes should include:
- Reinforcement Learning Toolbox
- Simulink
- Aerospace Blockset (recommended)
- Parallel Computing Toolbox (for GPU)

### Project Setup

1. **Open MATLAB** and navigate to the project directory:
   ```matlab
   cd /path/to/Parrot-with-PPO-Controller
   ```

2. **Initialize the project**:
   ```matlab
   % The project should auto-initialize
   % If not, manually run:
   cd tasks
   startVars
   cd ..
   ```

3. **Verify setup**:
   ```matlab
   % Check if variables are loaded
   who

   % Should see: vehicleVars, controllerVars, sensorsVars, etc.
   ```

---

## Training a New Agent

### Step 1: Prepare the Environment

1. **Open the main training script**:
   ```matlab
   open CreateParrotEnvironmantAndTrainAgent.mlx
   ```

2. **Choose your drone model**:
   ```matlab
   % For Parrot Mambo
   setMamboModel()

   % OR for Parrot Rolling Spider
   setRollingSpiderModel()
   ```

### Step 2: Configure Training Parameters

In the Live Script, modify these sections:

#### 2.1 Reward Function Parameters

```matlab
% Altitude error weight
w1 = 1.0;

% Velocity penalty weight
w2 = 0.1;

% Scaling factors
alpha = 0.5;  % Altitude error scaling
beta = 1.0;   % Velocity scaling
```

**Tips**:
- Increase `w1` to prioritize altitude accuracy
- Increase `w2` to reduce oscillations
- Adjust `alpha` and `beta` to change sensitivity

#### 2.2 Network Architecture

```matlab
% Hidden layer sizes
actorLayerSizes = [128, 128];
criticLayerSizes = [128, 128];

% Activation function
activationFcn = 'tanh';  % Options: 'tanh', 'relu', 'sigmoid'
```

**Tips**:
- Larger networks learn complex behaviors but train slower
- Start with [128, 128] as baseline
- Use 'tanh' for bounded outputs

#### 2.3 Training Options

```matlab
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...              % Total training episodes
    'MaxStepsPerEpisode', 2000, ...       % Steps per episode (100s @ 5ms)
    'ScoreAveragingWindowLength', 100, ...% Smoothing window
    'Verbose', true, ...                  % Show progress
    'Plots', 'training-progress', ...     % Display training plot
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -0.1, ...        % Stop when reward > -0.1
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', -0.5);              % Save agents with reward > -0.5
```

### Step 3: Create the Environment

```matlab
% Open the Simulink model
mdl = 'parrotMinidroneHover';
open_system(mdl);

% Define observation info
obsInfo = rlNumericSpec([6 1], ...
    'LowerLimit', -inf*ones(6,1), ...
    'UpperLimit', inf*ones(6,1));
obsInfo.Name = 'observations';
obsInfo.Description = 'altitude error, velocities, attitudes';

% Define action info
actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -1, ...
    'UpperLimit', 1);
actInfo.Name = 'thrust';

% Create environment
agentBlock = [mdl '/RL Agent'];
env = rlSimulinkEnv(mdl, agentBlock, obsInfo, actInfo);
```

### Step 4: Create the Agent

```matlab
% Create actor network
actorNetwork = createActorNetwork(obsInfo, actInfo, actorLayerSizes);

% Create critic network
criticNetwork = createCriticNetwork(obsInfo, criticLayerSizes);

% Create PPO agent
agent = rlPPOAgent(actorNetwork, criticNetwork);

% Set learning rates
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-4;

% Set PPO parameters
agent.AgentOptions.ClipFactor = 0.2;
agent.AgentOptions.EntropyLossWeight = 0.01;
agent.AgentOptions.MiniBatchSize = 128;
agent.AgentOptions.NumEpoch = 3;
agent.AgentOptions.ExperienceHorizon = 2048;
agent.AgentOptions.DiscountFactor = 0.99;
agent.AgentOptions.GAEFactor = 0.95;

% Enable GPU
agent.AgentOptions.UseDevice = 'gpu';
```

### Step 5: Start Training

```matlab
% Train the agent
trainingStats = train(agent, env, trainOpts);

% Training will display a real-time plot showing:
% - Episode reward
% - Episode Q0 (initial value estimate)
% - Average reward
```

### Step 6: Save the Trained Agent

```matlab
% Generate filename with date
dateStr = datestr(now, 'mmdd');
filename = sprintf('trainedAgent_%s.mat', dateStr);

% Save agent and training stats
save(filename, 'agent', 'trainingStats');

fprintf('Agent saved as: %s\n', filename);
```

**Training Time Estimate**:
- With GPU: 2-4 hours for 1000 episodes
- Without GPU: 10-20 hours

---

## Testing Pre-trained Agents

### Step 1: Load a Pre-trained Agent

```matlab
% Load one of the provided agents
load('1001.mat', 'agent');  % Best performing agent

% OR load your custom agent
% load('trainedAgent_1115.mat', 'agent');
```

### Step 2: Setup Simulation

```matlab
% Open the model
open_system('mainModels/parrotMinidroneHover');

% Set simulation time
set_param('parrotMinidroneHover', 'StopTime', '100');

% Configure logging
set_param('parrotMinidroneHover', 'SaveOutput', 'on');
```

### Step 3: Run Simulation

```matlab
% Method 1: Using sim() - Simplest
simOut = sim('parrotMinidroneHover');

% Method 2: Using RL simulation options - More control
simOptions = rlSimulationOptions('MaxSteps', 2000);
experience = sim(env, agent, simOptions);
```

### Step 4: Visualize Results

```matlab
% Extract logged data
altitude = simOut.logsout.get('altitude').Values;
time = altitude.Time;
alt_data = altitude.Data;

% Plot altitude tracking
figure;
plot(time, alt_data, 'b-', 'LineWidth', 2);
hold on;
plot([time(1), time(end)], [0, 0], 'r--', 'LineWidth', 1.5);  % Target
xlabel('Time (s)');
ylabel('Altitude (m)');
title('Altitude Tracking Performance');
legend('Actual', 'Target');
grid on;

% Calculate metrics
settling_idx = find(abs(alt_data) < 0.02*abs(alt_data(end)), 1);
settling_time = time(settling_idx);
overshoot = (max(alt_data) - 0) * 100;
steady_error = abs(mean(alt_data(end-100:end)));

fprintf('Performance Metrics:\n');
fprintf('  Settling Time: %.2f s\n', settling_time);
fprintf('  Overshoot: %.1f%%\n', overshoot);
fprintf('  Steady-State Error: %.3f m\n', steady_error);
```

---

## Evaluating Performance

### Comprehensive Evaluation

```matlab
% Run multiple episodes for statistical analysis
numEpisodes = 10;
results = struct();

for i = 1:numEpisodes
    fprintf('Running episode %d/%d...\n', i, numEpisodes);

    % Run simulation
    simOut = sim('parrotMinidroneHover');

    % Extract data
    altitude = simOut.logsout.get('altitude').Values;

    % Calculate metrics
    results(i).settling_time = calculateSettlingTime(altitude);
    results(i).overshoot = calculateOvershoot(altitude);
    results(i).rmse = calculateRMSE(altitude);
    results(i).steady_error = calculateSteadyError(altitude);
end

% Display statistics
fprintf('\nStatistical Summary (n=%d):\n', numEpisodes);
fprintf('  Settling Time: %.2f ± %.2f s\n', ...
    mean([results.settling_time]), std([results.settling_time]));
fprintf('  Overshoot: %.1f ± %.1f%%\n', ...
    mean([results.overshoot]), std([results.overshoot]));
fprintf('  RMSE: %.3f ± %.3f m\n', ...
    mean([results.rmse]), std([results.rmse]));
```

### Comparing Multiple Agents

```matlab
% Load multiple agents
agents = {
    load('0825.mat', 'agent'),
    load('0829.mat', 'agent'),
    load('0926.mat', 'agent'),
    load('1001.mat', 'agent')
};
agent_names = {'Aug 25', 'Aug 29', 'Sep 26', 'Oct 1'};

% Evaluate each
performance = zeros(length(agents), 3);  % [settling, overshoot, rmse]

for i = 1:length(agents)
    % Load agent
    agent = agents{i}.agent;

    % Simulate
    simOut = sim('parrotMinidroneHover');
    altitude = simOut.logsout.get('altitude').Values;

    % Store metrics
    performance(i, 1) = calculateSettlingTime(altitude);
    performance(i, 2) = calculateOvershoot(altitude);
    performance(i, 3) = calculateRMSE(altitude);
end

% Plot comparison
figure;
bar(performance);
set(gca, 'XTickLabel', agent_names);
ylabel('Performance Value');
legend('Settling Time (s)', 'Overshoot (%)', 'RMSE (m)');
title('Agent Performance Comparison');
grid on;
```

---

## Customizing the Environment

### Changing Target Altitude

```matlab
% In the Simulink model, modify the altitude reference block
% Or programmatically:
set_param('parrotMinidroneHover/Altitude_Reference', 'Value', '5');  % 5m target
```

### Adding Wind Disturbances

```matlab
% Enable wind in the environment
set_param('parrotMinidroneHover/Environment/Wind', 'Commented', 'off');

% Set wind parameters
windSpeed = 3;  % m/s
windDirection = 45;  % degrees
set_param('parrotMinidroneHover/Environment/Wind/Speed', 'Value', num2str(windSpeed));
set_param('parrotMinidroneHover/Environment/Wind/Direction', 'Value', num2str(windDirection));
```

### Modifying Sensor Noise

```matlab
% Increase IMU noise (in sensorsVars.m or directly)
sensorsVars.IMU.AccelNoise = 0.001;  % Increase from 0.0003924
sensorsVars.IMU.GyroNoise = 0.0003;  % Increase from 0.0001

% Reload variables
startVars;
```

### Using Different Drone Models

```matlab
% Custom drone parameters
customDrone.mass = 0.080;  % kg
customDrone.inertia = diag([8e-5, 1e-4, 1.5e-4]);  % kg⋅m²
customDrone.armLength = 0.06;  % m

% Load custom parameters
vehicleVars = customDrone;

% Re-initialize model
startVars;
```

---

## Hardware Deployment

### Step 1: Prepare for Code Generation

```matlab
% Ensure you have required toolboxes
% - Simulink Coder
% - Embedded Coder
% - Parrot Minidrone Support Package

% Check installation
ver('MATLAB Coder')
ver('Simulink Coder')
```

### Step 2: Configure Target Hardware

```matlab
% Open configuration parameters
open_system('parrotMinidroneHover');
set_param('parrotMinidroneHover', 'SystemTargetFile', 'ert.tlc');

% Set hardware board
set_param('parrotMinidroneHover', 'HardwareBoard', 'Parrot Minidrone');

% Configure code generation options
set_param('parrotMinidroneHover', 'GenerateReport', 'on');
set_param('parrotMinidroneHover', 'LaunchReport', 'on');
```

### Step 3: Generate and Deploy Code

```matlab
% Use the provided utility
cd utilities
generateFlightCode

% Or manually:
% Build the model
slbuild('parrotMinidroneHover');

% Deploy to drone (requires drone connection)
% Follow Parrot Support Package documentation
```

### Step 4: Safety Considerations

**Before deploying to hardware:**

1. **Test in simulation thoroughly**
2. **Start with low altitude targets** (0.5m)
3. **Have manual override ready**
4. **Use protective cage or net**
5. **Test in open area away from obstacles**
6. **Monitor battery level**
7. **Keep emergency stop accessible**

---

## Visualization Options

### Real-time Scopes

```matlab
% Enable scopes in the model
set_param('parrotMinidroneHover/Visualization/Scopes', 'Commented', 'off');

% Run simulation
sim('parrotMinidroneHover');
```

### 3D Animation

```matlab
% Enable Simulink 3D Animation
set_param('parrotMinidroneHover/Visualization/3D', 'Commented', 'off');

% Run simulation - 3D window will open automatically
sim('parrotMinidroneHover');
```

### FlightGear Integration

```matlab
% Install FlightGear (external application)
% https://www.flightgear.org/

% Configure FlightGear interface in Simulink
set_param('parrotMinidroneHover/Visualization/FlightGear', 'Commented', 'off');

% Start FlightGear first, then run simulation
sim('parrotMinidroneHover');
```

### Custom Plotting

```matlab
% Run simulation and collect data
simOut = sim('parrotMinidroneHover');

% Create comprehensive plots
figure('Position', [100 100 1200 800]);

% Subplot 1: Position
subplot(3,2,1);
plot(simOut.logsout.get('position').Values);
title('3D Position');
xlabel('Time (s)');
ylabel('Position (m)');
legend('X', 'Y', 'Z');
grid on;

% Subplot 2: Velocity
subplot(3,2,2);
plot(simOut.logsout.get('velocity').Values);
title('Velocity');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('u', 'v', 'w');
grid on;

% Subplot 3: Attitude
subplot(3,2,3);
attitude = simOut.logsout.get('attitude').Values;
plot(attitude.Time, rad2deg(attitude.Data));
title('Attitude Angles');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('Roll', 'Pitch', 'Yaw');
grid on;

% Subplot 4: Motor Commands
subplot(3,2,4);
plot(simOut.logsout.get('motors').Values);
title('Motor Commands');
xlabel('Time (s)');
ylabel('Command');
legend('M1', 'M2', 'M3', 'M4');
grid on;

% Subplot 5: Altitude Error
subplot(3,2,5);
altitude = simOut.logsout.get('altitude').Values;
target = 0;  % Target altitude
error = altitude.Data - target;
plot(altitude.Time, error);
title('Altitude Tracking Error');
xlabel('Time (s)');
ylabel('Error (m)');
grid on;

% Subplot 6: Reward
subplot(3,2,6);
plot(simOut.logsout.get('reward').Values);
title('Instantaneous Reward');
xlabel('Time (s)');
ylabel('Reward');
grid on;
```

---

## Advanced Usage

### Curriculum Learning

Train the agent with progressively harder tasks:

```matlab
% Stage 1: Easy - Small target altitude
targetAltitude = 1.0;  % 1m
agent = trainStage(agent, env, targetAltitude, 200);

% Stage 2: Medium - Moderate altitude
targetAltitude = 3.0;  % 3m
agent = trainStage(agent, env, targetAltitude, 300);

% Stage 3: Hard - Large altitude
targetAltitude = 5.0;  % 5m
agent = trainStage(agent, env, targetAltitude, 500);
```

### Domain Randomization

Train with varied parameters for robustness:

```matlab
for episode = 1:numEpisodes
    % Randomize mass (±10%)
    vehicleVars.mass = nominalMass * (0.9 + 0.2*rand());

    % Randomize inertia (±15%)
    vehicleVars.inertia = nominalInertia .* (0.85 + 0.3*rand(3,3));

    % Randomize sensor noise
    sensorsVars.IMU.AccelNoise = baseNoise * (0.5 + rand());

    % Train episode
    trainOneEpisode(agent, env);
end
```

### Transfer Learning

Adapt agent to new task:

```matlab
% Load pre-trained agent
load('1001.mat', 'agent');

% Freeze early layers (optional)
agent.Actor.NetworkParameters(1:4).LearnRateFactor = 0;

% Fine-tune on new task
newEnv = createNewEnvironment();  % E.g., trajectory tracking
agent = train(agent, newEnv, fineTuneOpts);
```

### Batch Evaluation

Evaluate across different conditions:

```matlab
% Define test conditions
testConditions = {
    struct('mass', 0.063, 'wind', 0),
    struct('mass', 0.070, 'wind', 0),
    struct('mass', 0.063, 'wind', 2),
    struct('mass', 0.070, 'wind', 2)
};

% Run tests
results = cell(length(testConditions), 1);
for i = 1:length(testConditions)
    % Apply condition
    vehicleVars.mass = testConditions{i}.mass;
    windSpeed = testConditions{i}.wind;

    % Simulate
    results{i} = evaluateAgent(agent, env);
end

% Analyze results
analyzeRobustness(results, testConditions);
```

---

## Helper Functions

### Custom Functions to Add to Your Workspace

```matlab
function settling_time = calculateSettlingTime(signal, target, threshold)
    % Calculate 2% settling time
    if nargin < 3
        threshold = 0.02;
    end
    if nargin < 2
        target = 0;
    end

    data = signal.Data;
    time = signal.Time;
    error = abs(data - target);
    final_value = abs(data(end) - target);

    idx = find(error > threshold * abs(final_value), 1, 'last');
    if isempty(idx)
        settling_time = time(1);
    else
        settling_time = time(idx);
    end
end

function overshoot = calculateOvershoot(signal, target)
    % Calculate overshoot percentage
    if nargin < 2
        target = 0;
    end

    data = signal.Data;
    overshoot = (max(data) - target) / abs(target) * 100;
end

function rmse = calculateRMSE(signal, target)
    % Calculate root mean square error
    if nargin < 2
        target = 0;
    end

    data = signal.Data;
    rmse = sqrt(mean((data - target).^2));
end

function steady_error = calculateSteadyError(signal, target)
    % Calculate steady-state error (last 10% of signal)
    if nargin < 2
        target = 0;
    end

    data = signal.Data;
    n = length(data);
    steady_idx = round(0.9*n):n;
    steady_error = abs(mean(data(steady_idx)) - target);
end
```

---

## Tips and Best Practices

### Training Tips

1. **Start Simple**: Begin with basic reward functions and small networks
2. **Monitor Training**: Watch the training plot for divergence or plateaus
3. **Save Checkpoints**: Save agents periodically during long training
4. **Use GPU**: Always enable GPU for faster training
5. **Adjust Learning Rates**: If training unstable, reduce learning rates by 10x

### Simulation Tips

1. **Check Initial Conditions**: Ensure drone starts in stable configuration
2. **Use Reasonable Targets**: Start with small altitude changes (< 2m)
3. **Enable Logging**: Always log important signals for analysis
4. **Run Multiple Episodes**: Single runs may not be representative

### Debugging Tips

1. **Check Observations**: Ensure observations are properly normalized
2. **Verify Actions**: Check action scaling and saturation limits
3. **Monitor Reward**: Ensure reward function gives expected values
4. **Inspect Gradients**: Use agent.getLearnableParameters() to check for NaN

---

## FAQs

**Q: Training is very slow. How can I speed it up?**
A: Ensure GPU is enabled, reduce network size, or use parallel workers.

**Q: Agent performs well in training but poorly in testing. Why?**
A: Likely overfitting. Add domain randomization or reduce network complexity.

**Q: Can I use this for position control, not just altitude?**
A: Yes, but you'll need to modify the observation and action spaces, and retrain.

**Q: How do I save intermediate agents during training?**
A: Set `SaveAgentCriteria` and `SaveAgentValue` in training options.

**Q: Can I train without a GPU?**
A: Yes, but it will be much slower. Set `UseDevice = 'cpu'`.

**Q: How do I resume training from a saved agent?**
A: Load the agent and call `train()` again with new training options.

---

## Additional Resources

- **MATLAB Documentation**: https://www.mathworks.com/help/reinforcement-learning/
- **Simulink Documentation**: https://www.mathworks.com/help/simulink/
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Project Repository**: [GitHub link]

---

**Guide Version**: 1.0
**Last Updated**: November 2025
