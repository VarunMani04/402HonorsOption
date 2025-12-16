# DS402 Honors Option: Neural Network Architectures for MDPs and Image Classification

This project implements and compares three distinct neural network architectures:
1. **CNN (Convolutional Neural Network)** - For image classification on CIFAR-10
2. **LSTM (Long Short-Term Memory)** - For sequential decision-making in Parking MDP
3. **Default/Standard Neural Network** - Baseline fully connected network for Parking MDP

## Overview

The goal of this project is to develop a plug-and-play framework where different neural network architectures can be easily tested on different problem domains:

- **CNN**: Trained on CIFAR-10 image classification task (32×32 RGB images, 10 classes)
- **LSTM**: Trained on Parking MDP for sequential reinforcement learning (121 states, 2 actions)
- **Default NN**: Trained on Parking MDP as a baseline comparison (121 states, 2 actions)

All networks are implemented in a unified framework using PyTorch and can be easily swapped via configuration parameters.

## Project Structure

```
Lab 5 - MDPs - Moving from Values to Policies with Policy Iteration/
├── agent/
│   ├── AgentBase.py              # Base agent class
│   ├── DeepQNetworkAgent.py      # Deep Q-Network agent with plug-and-play architectures
│   ├── QLearningAgent.py         # Tabular Q-learning agent
│   ├── PolicyIterationAgent.py   # Policy iteration agent
│   └── RandomAgent.py            # Random baseline agent
├── data/                         # Dataset files
│   ├── cifar-10-batches-py/     # CIFAR-10 dataset
│   ├── MDP1.txt, MDP2.txt, MDP3.txt  # MDP configuration files
├── comprehensive_network_analysis.py  # Main analysis script (runs all networks)
├── MDP.py                        # MDP class implementation
├── ParkingDefs.py                # Parking MDP definitions and enums
├── ParkingLotFactory.py          # Parking MDP factory
├── main.py                       # Original lab entry point
├── tests*.py                     # Lab test scripts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - PyTorch
   - NumPy
   - Matplotlib
   - Scikit-learn
   - SciPy
   - Torchvision

3. **Download CIFAR-10 dataset** (automatically downloaded on first run if not present)

## Usage

### Running All Networks and Analysis

To run all three networks and generate comprehensive analysis:

```bash
python comprehensive_network_analysis.py
```

This will:
- Train CNN on CIFAR-10 (20 epochs)
- Train LSTM on Parking MDP (50 epochs)
- Train Default NN on Parking MDP (50 epochs)
- Generate visualizations (learning curves, equivalence classes, action distribution, qualitative analysis)
- Save results to `analysis_results.pkl` for later analysis

**Note**: CNN training takes approximately 10-20 minutes depending on hardware.

### Running Individual Networks

You can also use the `ComprehensiveNetworkAnalyzer` class programmatically:

```python
from comprehensive_network_analysis import ComprehensiveNetworkAnalyzer

analyzer = ComprehensiveNetworkAnalyzer(seed_value=42)

# Train CNN on CIFAR-10
analyzer.train_cnn_on_cifar10(num_epochs=20, batch_size=64)

# Train LSTM on Parking MDP
analyzer.train_lstm_on_mdp(num_epochs=50, num_samples=30)

# Train Default NN on Parking MDP
analyzer.train_default_on_mdp(num_epochs=50, num_samples=30)

# Generate visualizations
analyzer.generate_visualizations()
```

### Using Networks Directly (Plug-and-Play)

You can create and use each network type directly via `DeepQNetworkAgent`:

```python
from agent.DeepQNetworkAgent import DeepQNetworkAgent
from ParkingLotFactory import createParkingMDP

# Create MDP
mdp, start = createParkingMDP("MyParkingMDP", busyRate=0.7)

# CNN Network
cnn_agent = DeepQNetworkAgent(
    "CNN_Agent", 
    mdp.numActions, 
    mdp.numStates,
    network_type='cnn',
    conv_channels=[32, 64, 128]
)

# LSTM Network
lstm_agent = DeepQNetworkAgent(
    "LSTM_Agent",
    mdp.numActions,
    mdp.numStates,
    network_type='lstm',
    hidden_size=64,
    num_layers=2
)

# Default/Standard Network
default_agent = DeepQNetworkAgent(
    "Default_Agent",
    mdp.numActions,
    mdp.numStates,
    network_type='standard',  # or omit (this is default)
    layer_sizes=[128, 64]
)

# Use agents with MDP
from testsDqnLab import trainingAndTestLoop
rewards = trainingAndTestLoop(50, 30, mdp, start, cnn_agent, maxTrajectoryLen=50)
```

### Command Line Options

```bash
# Skip CNN training (useful if CNN was already trained)
python comprehensive_network_analysis.py --skip-cnn

# Train only Default network
python comprehensive_network_analysis.py --default-only
```

## Network Architectures

### 1. CNN (Convolutional Neural Network)
- **Purpose**: Image classification on CIFAR-10
- **Architecture**: 
  - Convolutional layers: [32, 64, 128] channels
  - Kernel size: 3×3
  - Batch normalization: Enabled
  - Dropout: 0.5
  - Fully connected: 256 → 128 → 10
- **Input**: 32×32×3 RGB images
- **Output**: 10 class probabilities
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss

### 2. LSTM (Long Short-Term Memory)
- **Purpose**: Sequential decision-making in reinforcement learning
- **Architecture**:
  - LSTM layers: 2 layers, hidden size 64
  - Sequence length: 10
  - Multi-head attention: 4 heads
  - Fully connected: 64 → 32 → 2
  - Dropout: 0.3
- **Input**: One-hot encoded state (121 dimensions)
- **Output**: Q-values for 2 actions
- **Training**: Deep Q-Learning (DQN) with experience replay
- **Hyperparameters**: γ=0.7, ε=0.9, lr=0.001

### 3. Default/Standard Neural Network
- **Purpose**: Baseline fully connected network
- **Architecture**:
  - Fully connected layers: 121 → 128 → 64 → 2
  - Activation: ReLU
  - No dropout or batch normalization
- **Input**: One-hot encoded state (121 dimensions)
- **Output**: Q-values for 2 actions
- **Training**: Deep Q-Learning (DQN) with experience replay
- **Hyperparameters**: γ=0.7, ε=0.9, lr=0.001

## Parking MDP Environment

The Parking MDP simulates an agent navigating a parking lot:

- **States**: 121 (30 parking spaces × 4 states per space + 1 exit state)
  - Each space has 4 states: Parked, Crashed, Driving_Occupied, Driving_Available
- **Actions**: 2 (PARK, DRIVE)
- **Rewards**: 
  - +1000 (parked, decays with distance)
  - -10,000 (crashed)
  - -1 (waiting/driving)
- **Busy Rate**: 0.7 (70% occupancy probability, decays with space index)
- **Start State**: State 58 (Driving_Available in second-to-last space)

## Analysis Metrics

The analysis script generates comprehensive metrics for each network:

1. **Learning Curves**: Performance over training epochs
2. **Equivalence Classes**: Clustering of states with similar Q-values
3. **Action Distribution**: Uniformity of action selection
4. **Qualitative Analysis**: Q-value variance and spread patterns

All metrics are saved as visualizations and can be analyzed from the saved results.

## Outputs

Running the analysis script generates:

- **Visualizations** (PNG files):
  - `learning_curves_analysis.png`
  - `equivalence_classes_analysis.png`
  - `action_distribution_analysis.png`
  - `qualitative_analysis.png`
  - `comprehensive_network_analysis.png`

- **Data**:
  - `analysis_results.pkl` - Saved results for faster re-analysis and manual report generation

## Notes

- All networks use random seed 42 for reproducibility
- Training can be time-consuming (especially CNN). Results are cached in `analysis_results.pkl` to avoid retraining
- The CNN requires CIFAR-10 dataset (~170MB), which downloads automatically on first run
- MDP networks use experience replay buffer (size 10,000) for stable learning
- Results can be loaded from `analysis_results.pkl` to regenerate visualizations without retraining

