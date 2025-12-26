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
