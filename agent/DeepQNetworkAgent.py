from random import random, randrange, gauss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from agent.AgentBase import AgentBase, Verbosity
from ParkingDefs import StateType, Act
#Progress 
#Finished LSTM and CNN Newtowkrs
#.init functions -> COnstructor 
#CNN.forward -> convert CNN to 2D grids that bass throug layers (for MDP Data)
#LSTM.forward -> sequential input through LSTM layers 
#.get_sequence_input() -> converts the stored state into a sequence and pads with zeros
#ExperienceReplayBuffer -> Stores a single experience in a buffer and allows 
# the agent to learn from past experiences 
#Loss Function -> selects the approprate loss function based on problem 
# -> Cross-Entrophy: Classification, MSE -> Prediction, Huber -> Outliers 
class QNetworkBase(nn.Module, ABC):
    def __init__(self, input_size, output_size):
        super(QNetworkBase, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_device_compatible_input(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x

class QNetwork(QNetworkBase):
    """Standard fully connected Q-Network"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(QNetwork, self).__init__(input_size, output_size)
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.get_device_compatible_input(x)
        return self.network(x)

class CNNQNetwork(QNetworkBase):
    """CNN-based Q-Network for spatial feature extraction, compatible with CIFAR-10"""
    def __init__(self, input_size, output_size, conv_channels=[32, 64, 128], kernel_size=3, 
                 dropout_rate=0.5, use_batch_norm=True, input_channels=1):
        super(CNNQNetwork, self).__init__(input_size, output_size)
        
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Handle different input formats (1D state vector vs image data)
        if input_channels == 1:  # For MDP state vectors
            self.grid_size = int(np.sqrt(input_size))
            if self.grid_size * self.grid_size != input_size:
                self.grid_size = int(np.ceil(np.sqrt(input_size)))
                self.padded_size = self.grid_size * self.grid_size
            else:
                self.padded_size = input_size
        else:  # For image data (e.g., CIFAR-10: 32x32x3)
            self.grid_size = 32  # Standard CIFAR-10 size
            self.padded_size = input_size
        
        # Enhanced CNN layers for better feature extraction
        layers = []
        in_channels = input_channels
        for i, out_channels in enumerate(conv_channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate) if i > 0 else nn.Identity()  # No dropout on first layer
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flattened_size = self._get_conv_output_size()
        
        # Enhanced fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size)
        )
    
    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.grid_size, self.grid_size)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.size()))
    
    def forward(self, x):
        x = self.get_device_compatible_input(x) 
        batch_size = x.size(0)
        
        # Handle different input types
        if self.input_channels == 1:
            # For MDP state vectors - reshape to 2D grid for spatial processing
            if x.size(1) < self.padded_size:
                padding = torch.zeros(batch_size, self.padded_size - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            x = x.view(batch_size, 1, self.grid_size, self.grid_size)
        else:
            # For image data (already in correct format)
            pass
        
        # Pass through CNN layers
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        # Pass through fully connected layers
        return self.fc_layers(x)

class LSTMQNetwork(QNetworkBase):
    """LSTM-based Q-Network for sequential MDP planning problems"""
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, 
                 dropout_rate=0.3, sequence_length=10):
        super(LSTMQNetwork, self).__init__(input_size, output_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Enhanced LSTM with dropout for better generalization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Attention mechanism for better sequence processing
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout_rate)
        
        # Enhanced output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # State history buffer for sequence processing
        self.state_history = deque(maxlen=sequence_length)
        
    def forward(self, x):
        x = self.get_device_compatible_input(x)
        batch_size = x.size(0)
        
        # Handle sequence input for MDP planning
        if x.size(1) == self.input_size:
            # Single state input - create sequence from history
            x = x.unsqueeze(1)  # Add sequence dimension
            if len(x.size()) == 2:
                x = x.unsqueeze(1)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Apply attention if sequence is long enough
        if lstm_out.size(1) > 1:
            # Reshape for attention: (seq_len, batch, hidden_size)
            lstm_out_transposed = lstm_out.transpose(0, 1)
            attended_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
            # Take the last attended output
            final_output = attended_out[-1]  # Last time step
        else:
            # Single time step - just take the output
            final_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        return self.fc_layers(final_output)
    
    def update_state_history(self, state):
        """Update state history for sequence processing"""
        self.state_history.append(state)
    
    def get_sequence_input(self, current_state):
        """Create sequence input from state history"""
        if len(self.state_history) == 0:
            return torch.zeros(1, 1, self.input_size)
        
        # Convert history to tensor sequence
        history_list = list(self.state_history)
        if len(history_list) < self.sequence_length:
            # Pad with zeros if not enough history
            padding_needed = self.sequence_length - len(history_list)
            history_list = [0] * padding_needed + history_list
        
        return torch.tensor(history_list, dtype=torch.float32).unsqueeze(0)

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class LossFunction:
    @staticmethod
    def get_loss_fn(network_type, problem_type='reinforcement'):
        if network_type.lower() == 'cnn' and problem_type == 'classification':
            return nn.CrossEntropyLoss()
        elif network_type.lower() == 'lstm' and problem_type == 'sequence':
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
    
    @staticmethod
    def huber_loss(predicted, target, delta=1.0):
        """Huber loss for robust Q-learning"""
        error = predicted - target
        is_small_error = torch.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * torch.abs(error) - 0.5 * delta ** 2
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

#Next steps: 
#Step 1: 
#backwards pass: pick a loss function and this depends on the problem
#loss function can be MRSE, RMSE, cross entrophy loss

#look into supervised vs reinforcment learning 
#Step 2:
#Fitting Options 
#Reinforcnment Learning Framing -> Link to the models 
#CNN: Look for image data CIFAR 10 (Object recognition dataset, smaller and size), MDAS Dataset
#CIFAR utilized in other dataset
#LSTM: Sequence Model: Planning problems in previous labs (classification)
#Traditionally utilized for language problems, time-series data  


class DeepQNetworkAgent(AgentBase):
    #spatial vs temporal
    def __init__(self, name, numActions, numStates, probGreedy=.9, discountFactor=.7, learningRate=.0001, 
                 layer_sizes=[128], regularization=.0001, seed=10, verbosity=Verbosity.SILENT, 
                 network_type='standard', loss_type='mse', use_replay_buffer=True, buffer_size=10000,
                 batch_size=32, **network_kwargs):
        super().__init__(name, verbosity)
        self.probGreedy = probGreedy
        self.evaluating = False
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.numActions = numActions
        self.numStates = numStates
        self.network_type = network_type
        self.batch_size = batch_size
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize PyTorch neural network with plug-and-play architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = self._create_network(network_type, layer_sizes, **network_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate, weight_decay=regularization)
        
        # Initialize loss function based on network type
        self.criterion = self._get_loss_function(loss_type)
        
        # Initialize experience replay buffer
        self.use_replay_buffer = use_replay_buffer
        if use_replay_buffer:
            self.replay_buffer = ExperienceReplayBuffer(buffer_size)
    
    def _create_network(self, network_type, layer_sizes, **kwargs):
        """Factory method for plug-and-play network architectures"""
        if network_type.lower() == 'cnn':
            return CNNQNetwork(self.numStates, self.numActions, 
                             conv_channels=kwargs.get('conv_channels', [32, 64]),
                             kernel_size=kwargs.get('kernel_size', 3))
        elif network_type.lower() == 'lstm':
            return LSTMQNetwork(self.numStates, self.numActions,
                              hidden_size=kwargs.get('hidden_size', 64),
                              num_layers=kwargs.get('num_layers', 2))
        else:  # default to standard fully connected
            return QNetwork(self.numStates, layer_sizes, self.numActions)
    
    def _get_loss_function(self, loss_type):
        """Get appropriate loss function based on type"""
        if loss_type.lower() == 'huber':
            return lambda pred, target: LossFunction.huber_loss(pred, target)
        elif loss_type.lower() == 'cross_entropy' and self.network_type in ['cnn', 'lstm']:
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def predict_qvalues(self, state): 
        state_onehot = torch.zeros(1, self.numStates, device=self.device) #converts state number into vector
        state_onehot[0][state] = 1
        
        with torch.no_grad(): #disable gradient calculation and creates a numpy array
            return self.model(state_onehot).cpu().numpy()[0]

    def findHighestValuedAction(self, state): #get's highest q value and returns the highest state
        q_values = self.predict_qvalues(state)
        return np.argmax(q_values)

    def selectAction(self, state, iteration, mdp): 
        if self.evaluating or random() < self.probGreedy:
            return self.findHighestValuedAction(state)
        else:
            return randrange(self.numActions)

    def observeReward(self, iteration, currentState, nextState, action, totalReward, rewardHere):
        if not self.evaluating:
            # Store experience in replay buffer if using it
            if self.use_replay_buffer:
                done = False  # Assume not terminal for this MDP
                self.replay_buffer.push(currentState, action, rewardHere, nextState, done)
                
                # Train from replay buffer if enough experiences
                if len(self.replay_buffer) >= self.batch_size:
                    self._replay_train()
            else:
                # Direct training without replay buffer
                self._train_single_step(currentState, action, rewardHere, nextState)

        return super().observeReward(iteration, currentState, nextState, action, totalReward, rewardHere)
    
    def _train_single_step(self, currentState, action, rewardHere, nextState):
        """Train on a single experience"""
        current_q_values = self.predict_qvalues(currentState)
        next_q_values = self.predict_qvalues(nextState)
        
        # Bellman equation
        bestNextAction = np.argmax(next_q_values)
        target_q_value = rewardHere + self.discountFactor * next_q_values[bestNextAction]
        
        # Prepare training data
        state_onehot = torch.zeros(1, self.numStates, device=self.device)
        state_onehot[0][currentState] = 1
        
        target_q_values = torch.tensor([current_q_values], dtype=torch.float32, device=self.device)
        target_q_values[0][action] = target_q_value
        
        # Train the model
        self.optimizer.zero_grad()
        predicted_q_values = self.model(state_onehot)
        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
    
    def _replay_train(self):
        """Train on a batch from replay buffer"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.zeros(self.batch_size, self.numStates, device=self.device)
        next_state_batch = torch.zeros(self.batch_size, self.numStates, device=self.device)
        
        for i, (state, next_state) in enumerate(zip(states, next_states)):
            state_batch[i][state] = 1
            next_state_batch[i][next_state] = 1
        
        action_batch = torch.tensor(actions, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Compute Q values
        current_q_values = self.model(state_batch)
        next_q_values = self.model(next_state_batch)
        
        # Compute target Q values
        target_q_values = current_q_values.clone()
        for i in range(self.batch_size):
            if done_batch[i]:
                target_q_values[i][action_batch[i]] = reward_batch[i]
            else:
                target_q_values[i][action_batch[i]] = reward_batch[i] + self.discountFactor * torch.max(next_q_values[i])
        
        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def setLearningRate(self, iteration, totalIterations):
        # Implement learning rate decay
        decay_rate = 0.95
        new_lr = self.learningRate * (decay_rate ** (iteration / 100))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def loadModelFromPickle(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def saveModelToPickle(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def determine_criticalities_huang(self):
        criticalities = []
        for state in range(self.numStates):
            q_values = self.predict_qvalues(state)
            crit = max(q_values) - np.average(q_values)
            criticalities.append((state, crit))
        return criticalities

    def determine_criticalities_amir(self):
        criticalities = []
        for state in range(self.numStates):
            q_values = self.predict_qvalues(state)
            crit = max(q_values) - min(q_values)
            criticalities.append((state, crit))
        return criticalities

    def printCriticalities(self, criticalities, parkingMode=False, numRows=2, numSpacesPerRow=10):
        if not parkingMode:
            print(criticalities)
        else:
            criticalities.sort(key=lambda tup: tup[1], reverse=True)
            for state, crit in criticalities:
                row, space = StateType.getIndices(state, numSpacesPerRow)
                print(row, "\t", space, "\t", StateType.get(state), "\t", crit)

    #FIXME identify a critical state in the toy MDP and parking MDP, do you agree (same task for non-critical)
    #FIXME have them write their own criticality function for the Q-learning agent
    #FIXME have them use criticality to "test" by ranking a set of agents arising from too-early stopping, mutation, and principled
    '''
    def mutate(self, variance): #might need to fix this
        for layer in range(len(self.model.coefs_)):
            for listIdx in range(len(self.model.coefs_[layer])):
                for coefIdx in range(len(self.model.coefs_[layer][listIdx])):
                    self.model.coefs_[layer][listIdx][coefIdx] = gauss(0, variance)*self.model.coefs_[layer][listIdx][coefIdx]
    '''
    # This function is peculiar to the parking MDP domain, everything else provided here is general to table-based Q-learning
    def analyzeQfn(self):
        numSpacesDeclined = 0
        for state in range(self.numStates):
            stateClass = StateType.get(state)
            if StateType.DRIVING_AVAILABLE == stateClass:
                q_values = self.predict_qvalues(state)
                if q_values[Act.PARK.value] < q_values[Act.DRIVE.value]:
                    numSpacesDeclined += 1
        result = self.name + "'s final Q function declines to park in unoccupied spaces " + str(numSpacesDeclined) + " times"
        return result

    def __repr__(self):
        result = "\nLearning Rate: " + str(self.learningRate) + "\nValue Function\n\t\t"

        for i in range(self.numActions):
            result += 'Action{:>3}'.format(chr(i+ord('A'))) + "\t"
        result += "\n"

        for i in range(self.numStates):
            result += 'State{:>3}'.format(i) + ":\t"
            q_values = self.predict_qvalues(i)
            for q_value in q_values:
                formattedNum = "{: 7.1f}".format(q_value)
                if len(formattedNum) > 8:  # string length is equal to the length of the scientific notation formatting string on next lines
                    formattedNum = "{:.1E}".format(q_value)
                result += '{:>8}'.format(formattedNum) + "\t"
            result += "\n"

        result += "\nPolicy\n"
        for i in range(self.numStates):
            result += 'State{:>3}'.format(i) + ":\t"
            result += chr(self.findHighestValuedAction(i) + ord('A'))
            result += "\n"
        return result


