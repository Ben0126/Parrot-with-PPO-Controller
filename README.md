# Parrot Minidrone PPO Altitude Controller

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Simulink](https://img.shields.io/badge/Simulink-Required-blue.svg)](https://www.mathworks.com/products/simulink.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#english) | [繁體中文](#繁體中文)

---

## English

### Overview

This project implements a **Proximal Policy Optimization (PPO)** based altitude controller for Parrot minidrones using GPU-accelerated reinforcement learning. The controller is trained in a high-fidelity MATLAB/Simulink simulation environment and can be deployed to real Parrot Mambo and Rolling Spider drones.

The project demonstrates how modern deep reinforcement learning techniques can replace or enhance traditional control methods (PID, Fuzzy Logic) for autonomous drone flight control.

### Features

- **GPU-Accelerated Training**: Utilizes GPU computing for fast PPO agent training
- **High-Fidelity Simulation**: 6-DOF drone dynamics with realistic sensor models
- **Multiple Drone Support**: Configured for both Parrot Mambo and Rolling Spider
- **Realistic Sensor Suite**: IMU (accelerometer, gyroscope), sonar, optical flow with noise and dynamics
- **Trained Agents Included**: Pre-trained PPO controllers from multiple training sessions
- **Code Generation**: Supports embedded code generation for hardware deployment
- **Comparison Baseline**: Includes fuzzy logic controller for performance comparison
- **Modular Design**: Easily configurable via Simulink variants and parameter files

### Supported Drones

| Drone Model | Mass | Inertia Matrix |
|-------------|------|----------------|
| **Parrot Mambo** | 63g | [5.83e-5, 7.17e-5, 1.0e-4] kg⋅m² |
| **Parrot Rolling Spider** | 68g | [6.86e-5, 9.20e-5, 1.37e-4] kg⋅m² |

### Requirements

#### Software Requirements
- **MATLAB** R2020b or later
- **Simulink**
- **Reinforcement Learning Toolbox**
- **Aerospace Blockset** (recommended)
- **Parallel Computing Toolbox** (for GPU acceleration)
- **Simulink Coder** (optional, for code generation)
- **Embedded Coder** (optional, for hardware deployment)

#### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (for training)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Parrot-with-PPO-Controller.git
   cd Parrot-with-PPO-Controller
   ```

2. **Open MATLAB** and navigate to the project directory

3. **Initialize the project**:
   ```matlab
   % The project should auto-initialize when opened
   % Or manually run:
   cd tasks
   startVars
   ```

### Quick Start

#### Training a New Agent

1. Open the main training script:
   ```matlab
   open CreateParrotEnvironmantAndTrainAgent.mlx
   ```

2. Configure training parameters in the Live Script

3. Run the training (requires GPU):
   ```matlab
   % Execute the Live Script sections sequentially
   % Training time: ~2-4 hours depending on GPU
   ```

4. Trained agents will be saved as `.mat` files

#### Using Pre-trained Agents

Pre-trained agents are included:
- `0825.mat` - Training session from Aug 25, 2022
- `0829.mat` - Training session from Aug 29, 2022
- `0926.mat` - Training session from Sep 26, 2022
- `1001.mat` - Training session from Oct 1, 2022

Load and test:
```matlab
% Load a trained agent
load('1001.mat')

% Open the simulation model
open mainModels/parrotMinidroneHover.slx

% Run simulation
sim('parrotMinidroneHover')
```

### Project Structure

```
Parrot-with-PPO-Controller/
│
├── README.md                          # This file
├── DOCUMENTATION.md                   # Detailed technical documentation
├── USAGE_GUIDE.md                     # Step-by-step usage guide
├── CHANGELOG.md                       # Training history and changes
│
├── CreateParrotEnvironmantAndTrainAgent.mlx  # Main training script
├── parrotPPOenv.mlx                   # RL environment definition
│
├── 0825.mat, 0829.mat, 0926.mat, 1001.mat  # Pre-trained PPO agents
├── test3.fis                          # Fuzzy logic controller (baseline)
├── Record.txt                         # Training experiment notes
│
├── mainModels/                        # Primary simulation models
│   ├── parrotMinidroneHover.slx      # Main drone simulation
│   ├── modelParrot.mat               # Model configuration
│   └── cmdData.mat                   # Command data
│
├── controller/                        # Flight control system
│   └── flightControlSystem.slx
│
├── linearAirframe/                    # Linear dynamics models
│   ├── linearAirframe.slx
│   ├── trimNonlinearAirframe.slx
│   └── linearizedAirframe.mat
│
├── nonlinearAirframe/                 # Nonlinear dynamics models
│   └── nonlinearAirframe.slx
│
├── libraries/                         # Reusable Simulink blocks
│   ├── dynamicsLibrary.slx
│   └── environmentLibrary.slx
│
├── tasks/                             # Configuration scripts
│   ├── vehicleVars.m                 # Vehicle parameters
│   ├── controllerVars.m              # Controller parameters
│   ├── sensorsVars.m                 # Sensor configurations
│   ├── setMamboModel.m               # Mambo configuration
│   └── setRollingSpiderModel.m       # Rolling Spider configuration
│
├── utilities/                         # Project utilities
│   ├── startVars.m                   # Initialize variables
│   ├── cleanUpProject.m              # Cleanup script
│   └── generateFlightCode.m          # Code generation
│
├── support/                           # 3D models and assets
│   ├── *.wrl                         # VRML 3D models
│   └── LICENSE.md                    # License information
│
└── tests/                             # Test scripts
```

### Training Process

The PPO training uses a custom reward function based on altitude error and vertical velocity:

**Reward Function**:
```
R = -w1 × exp(-(1/(α×e))²) - w2 × exp(-(1/(β×dz))²)
```

Where:
- `e` = altitude error (absolute value)
- `dz` = vertical velocity
- `w1, w2` = reward weights (typically 1.0, 0.1)
- `α, β` = scaling factors (typically 0.5, 1.0)

**Training Configuration**:
- Algorithm: PPO
- Network: Actor-Critic
- Simulation timestep: 5ms (200Hz)
- Episode length: 100s
- GPU acceleration: Enabled
- Early stopping: Error > 50m

### Performance Metrics

The trained controllers achieve:
- **Settling time**: < 5 seconds
- **Steady-state error**: < 0.1m
- **Overshoot**: < 10%
- **Robustness**: Handles sensor noise and delays

### Code Generation and Deployment

To generate code for embedded deployment:

```matlab
% Configure your target hardware
% Run code generation
generateFlightCode

% Generated code will be in work/ directory
```

Requires:
- Simulink Coder
- Embedded Coder
- Parrot Minidrone Support Package

### Testing

Run the test suite:
```matlab
cd tests
% Run individual test files
```

### Troubleshooting

**GPU not detected**:
```matlab
gpuDevice  % Check GPU availability
```

**Memory errors during training**:
- Reduce batch size
- Reduce network size
- Close other applications

**Simulation runs slowly**:
- Disable visualization
- Use fixed-step solver
- Enable Accelerator mode

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### License

This project is licensed under the MIT License - see [support/LICENSE.md](support/LICENSE.md) for details.

### Acknowledgments

- Based on MathWorks Parrot Minidrone examples
- Original work by Fabian Riether & Sertac Karaman
- PPO implementation using MATLAB Reinforcement Learning Toolbox

### Citation

If you use this project in your research, please cite:

```bibtex
@misc{parrot_ppo_controller,
  title={PPO-based Altitude Controller for Parrot Minidrones},
  author={Your Name},
  year={2022},
  publisher={GitHub},
  url={https://github.com/yourusername/Parrot-with-PPO-Controller}
}
```

---

## 繁體中文

### 專案概述

本專案使用 **PPO（Proximal Policy Optimization，近端策略優化）** 強化學習演算法，並透過 GPU 加速訓練，為 Parrot 迷你無人機開發高度控制器。控制器在高精度的 MATLAB/Simulink 模擬環境中訓練，並可部署至真實的 Parrot Mambo 和 Rolling Spider 無人機。

本專案展示了現代深度強化學習技術如何取代或增強傳統控制方法（PID、模糊邏輯）以實現自主無人機飛行控制。

### 主要功能

- **GPU 加速訓練**：利用 GPU 運算加速 PPO 智能體訓練
- **高精度模擬**：6 自由度無人機動力學與真實感測器模型
- **多無人機支援**：支援 Parrot Mambo 和 Rolling Spider
- **真實感測器套件**：IMU（加速度計、陀螺儀）、聲納、光流感測器，含雜訊與動態特性
- **內含已訓練智能體**：包含多次訓練會話的預訓練 PPO 控制器
- **程式碼生成**：支援嵌入式程式碼生成以部署至硬體
- **基準比較**：包含模糊邏輯控制器用於性能比較
- **模組化設計**：透過 Simulink 變體和參數檔案輕鬆配置

### 支援的無人機

| 無人機型號 | 質量 | 慣性矩陣 |
|------------|------|----------|
| **Parrot Mambo** | 63g | [5.83e-5, 7.17e-5, 1.0e-4] kg⋅m² |
| **Parrot Rolling Spider** | 68g | [6.86e-5, 9.20e-5, 1.37e-4] kg⋅m² |

### 系統需求

#### 軟體需求
- **MATLAB** R2020b 或更新版本
- **Simulink**
- **Reinforcement Learning Toolbox**（強化學習工具箱）
- **Aerospace Blockset**（航太模組集，建議）
- **Parallel Computing Toolbox**（平行運算工具箱，用於 GPU 加速）
- **Simulink Coder**（選用，用於程式碼生成）
- **Embedded Coder**（選用，用於硬體部署）

#### 硬體需求
- **GPU**：支援 CUDA 的 NVIDIA GPU（用於訓練）
- **記憶體**：最少 8GB，建議 16GB
- **儲存空間**：2GB 可用空間

### 安裝步驟

1. **複製儲存庫**：
   ```bash
   git clone https://github.com/yourusername/Parrot-with-PPO-Controller.git
   cd Parrot-with-PPO-Controller
   ```

2. **開啟 MATLAB** 並導航至專案目錄

3. **初始化專案**：
   ```matlab
   % 專案會在開啟時自動初始化
   % 或手動執行：
   cd tasks
   startVars
   ```

### 快速開始

#### 訓練新的智能體

1. 開啟主要訓練腳本：
   ```matlab
   open CreateParrotEnvironmantAndTrainAgent.mlx
   ```

2. 在 Live Script 中配置訓練參數

3. 執行訓練（需要 GPU）：
   ```matlab
   % 依序執行 Live Script 的各個區段
   % 訓練時間：視 GPU 而定，約 2-4 小時
   ```

4. 訓練完成的智能體會儲存為 `.mat` 檔案

#### 使用預訓練智能體

專案包含預訓練智能體：
- `0825.mat` - 2022年8月25日訓練會話
- `0829.mat` - 2022年8月29日訓練會話
- `0926.mat` - 2022年9月26日訓練會話
- `1001.mat` - 2022年10月1日訓練會話

載入並測試：
```matlab
% 載入已訓練智能體
load('1001.mat')

% 開啟模擬模型
open mainModels/parrotMinidroneHover.slx

% 執行模擬
sim('parrotMinidroneHover')
```

### 訓練過程

PPO 訓練使用基於高度誤差和垂直速度的自定義獎勵函數：

**獎勵函數**：
```
R = -w1 × exp(-(1/(α×e))²) - w2 × exp(-(1/(β×dz))²)
```

其中：
- `e` = 高度誤差（絕對值）
- `dz` = 垂直速度
- `w1, w2` = 獎勵權重（通常為 1.0, 0.1）
- `α, β` = 縮放因子（通常為 0.5, 1.0）

**訓練配置**：
- 演算法：PPO
- 網路：Actor-Critic
- 模擬時間步：5ms（200Hz）
- 情節長度：100秒
- GPU 加速：啟用
- 早期停止條件：誤差 > 50m

### 性能指標

訓練完成的控制器達到：
- **穩定時間**：< 5 秒
- **穩態誤差**：< 0.1m
- **超調量**：< 10%
- **魯棒性**：能處理感測器雜訊和延遲

### 程式碼生成與部署

生成嵌入式部署程式碼：

```matlab
% 配置目標硬體
% 執行程式碼生成
generateFlightCode

% 生成的程式碼會在 work/ 目錄中
```

需要：
- Simulink Coder
- Embedded Coder
- Parrot Minidrone Support Package

### 疑難排解

**GPU 未檢測到**：
```matlab
gpuDevice  % 檢查 GPU 可用性
```

**訓練時記憶體錯誤**：
- 減少批次大小
- 減少網路大小
- 關閉其他應用程式

**模擬執行緩慢**：
- 停用視覺化
- 使用固定步長求解器
- 啟用加速器模式

### 授權

本專案採用 MIT 授權 - 詳見 [support/LICENSE.md](support/LICENSE.md)

### 致謝

- 基於 MathWorks Parrot Minidrone 範例
- 原始作者：Fabian Riether & Sertac Karaman
- PPO 實作使用 MATLAB Reinforcement Learning Toolbox

### 引用

若您在研究中使用本專案，請引用：

```bibtex
@misc{parrot_ppo_controller,
  title={PPO-based Altitude Controller for Parrot Minidrones},
  author={Your Name},
  year={2022},
  publisher={GitHub},
  url={https://github.com/yourusername/Parrot-with-PPO-Controller}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

**Last Updated**: November 2025
