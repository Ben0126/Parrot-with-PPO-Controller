# Parrot Minidrone PPO Controller — training scripts and 2022 iterations (simulation only)

Undergraduate research code from the Intelligent Control Lab, National Taitung University (2022).
This repository holds the **training script and the 2022 development iterations** of a PPO altitude
controller for the Parrot minidrone. Everything runs **in MATLAB/Simulink simulation**.

**The cleaned-up version that goes with the papers is
[`Parrot-with-PPO-altitude-controller`](https://github.com/Ben0126/Parrot-with-PPO-altitude-controller)** —
start there for what the project is, my part in it, and the publication list.

## What is here

- `CreateParrotEnvironmantAndTrainAgent.mlx` — builds the Simulink RL environment and trains the PPO agent
  (MATLAB Reinforcement Learning Toolbox):
  - observation: 2-D (altitude error, vertical velocity);
  - action: 1-D thrust correction bounded to ±10;
  - actor 2-32-64-64-16-2 and critic 2-32-64-64-8-1, ReLU.
- `parrotPPOenv.mlx` — environment reset function (randomized initial condition).
- `controller/flightControlSystem.slx` — flight control system with the *Altitude Controller* subsystem
  that hosts the RL Agent block.
- `Record.txt` — my log of the reward-function variants.
- `test3.fis` — fuzzy-logic file used in the reward / controller experiments.
- `*.mat` — saved agents and run data from Aug–Oct 2022 (filenames are dates).

The base airframe, sensor and environment models and the project structure come from MathWorks'
*Parrot Minidrone Hover* example (Aerospace Blockset) and keep their original copyright.

## My part

I designed the RL formulation of the altitude controller: the observation (input), the action (output)
and the reward function, and iterated on it from April to October 2022 (network sizes, GAE, entropy
weight, reward shaping). Results are
reported in the co-authored papers listed in the companion repository; this repository has no
evaluation logs.

## Running it

MATLAB R2021a or later with Simulink, Aerospace Blockset, Reinforcement Learning Toolbox and Fuzzy Logic
Toolbox. Open `parrotMinidroneHover.prj`, then run `CreateParrotEnvironmantAndTrainAgent.mlx`.
