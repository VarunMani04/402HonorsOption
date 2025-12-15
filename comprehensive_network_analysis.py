#!/usr/bin/env python3
"""
Comprehensive Neural Network Analysis for Lab 5
- CNN: CIFAR-10 image classification
- LSTM: Parking MDP (sequential planning)
- Default/Standard NN: Parking MDP (baseline)

Metrics calculated:
1. Equivalence classes between parking states
2. Qualitative analysis
3. Entropy
4. Learning Curves
5. Uniformness of action distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from random import seed
from scipy.stats import entropy as scipy_entropy
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import json
import pickle
from datetime import datetime

# Lab 5 imports
from ParkingLotFactory import createParkingMDP
from agent.DeepQNetworkAgent import DeepQNetworkAgent, CNNQNetwork
from agent.AgentBase import Verbosity
from testsDqnLab import trainingAndTestLoop
from ParkingDefs import StateType, Act

# Use default style (seaborn may not be available)
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
np.set_printoptions(precision=4, suppress=True)


class ComprehensiveNetworkAnalyzer:
    """Comprehensive analysis of all three neural network architectures"""
    
    def __init__(self, seed_value=42):
        self.seed_value = seed_value
        seed(seed_value)
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        
        self.results = {}
        self.report_data = {
            'methodology': {},
            'equivalence_classes': {},
            'qualitative_analysis': {},
            'entropy': {},
            'learning_curves': {},
            'action_distribution': {}
        }
    
    def train_cnn_on_cifar10(self, num_epochs=20, batch_size=64):
        """Train CNN on CIFAR-10 dataset"""
        print("\nTraining CNN on CIFAR-10")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_test)
        
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Initialize model
        model = CNNQNetwork(
            input_size=32*32*3,
            output_size=10,  # 10 CIFAR-10 classes
            conv_channels=[32, 64, 128],
            kernel_size=3,
            dropout_rate=0.5,
            use_batch_norm=True,
            input_channels=3
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        test_losses = []
        
        print(f"Training on device: {device}")
        print(f"Training samples: {len(trainset)}, Test samples: {len(testset)}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(trainloader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(trainloader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
            train_loss = running_loss / len(trainloader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Testing phase
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in testloader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    test_loss += criterion(outputs, targets).item()
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_loss /= len(testloader)
            test_acc = 100. * correct / total
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Calculate action distribution (class predictions)
        model.eval()
        class_predictions = defaultdict(int)
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in testloader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                
                for pred in predicted.cpu().numpy():
                    class_predictions[pred] += 1
                    total_samples += 1
        
        # Store results
        self.results['CNN_CIFAR10'] = {
            'model': model,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'test_losses': test_losses,
            'final_train_acc': train_accuracies[-1],
            'final_test_acc': test_accuracies[-1],
            'class_distribution': dict(class_predictions),
            'total_samples': total_samples,
            'num_epochs': num_epochs
        }
        
        print(f"\nCNN Training Complete. Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        
        return model, train_accuracies, test_accuracies
    
    def train_lstm_on_mdp(self, num_epochs=50, num_samples=30):
        """Train LSTM on Parking MDP"""
        print("\nTraining LSTM on Parking MDP")
        
        mdp, start = createParkingMDP("LSTM_ParkingMDP", busyRate=0.7)
        
        print(f"MDP: {mdp.numStates} states, {mdp.numActions} actions")
        
        agent = DeepQNetworkAgent(
            "LSTM_Parking",
            mdp.numActions,
            mdp.numStates,
            network_type='lstm',
            hidden_size=64,
            num_layers=2,
            sequence_length=10,
            learningRate=0.001,
            seed=self.seed_value,
            verbosity=Verbosity.SILENT
        )
        
        rewards = trainingAndTestLoop(num_epochs, num_samples, mdp, start, agent, maxTrajectoryLen=50)
        
        # Analyze action distribution
        action_distribution = self._calculate_action_distribution(agent, mdp)
        
        # Calculate policy entropy
        policy_entropy = self._calculate_policy_entropy(agent, mdp)
        
        # Equivalence classes
        equiv_classes = self._analyze_equivalence_classes(agent, mdp, "LSTM")
        
        self.results['LSTM_Parking'] = {
            'agent': agent,
            'rewards': rewards,
            'mdp': mdp,
            'start': start,
            'action_distribution': action_distribution,
            'policy_entropy': policy_entropy,
            'equivalence_classes': equiv_classes,
            'final_reward': np.mean(rewards[-10:]),
            'num_epochs': num_epochs
        }
        
        print(f"\nLSTM Training Complete. Final Average Reward: {np.mean(rewards[-10:]):.2f}")
        
        return agent, rewards, mdp
    
    def train_default_on_mdp(self, num_epochs=50, num_samples=30):
        """Train Default/Standard NN on Parking MDP"""
        print("\nTraining Default/Standard NN on Parking MDP")
        
        mdp, start = createParkingMDP("Default_ParkingMDP", busyRate=0.7)
        
        print(f"MDP: {mdp.numStates} states, {mdp.numActions} actions")
        
        agent = DeepQNetworkAgent(
            "Default_Parking",
            mdp.numActions,
            mdp.numStates,
            network_type='standard',
            layer_sizes=[128, 64],
            learningRate=0.001,
            seed=self.seed_value,
            verbosity=Verbosity.SILENT
        )
        
        rewards = trainingAndTestLoop(num_epochs, num_samples, mdp, start, agent, maxTrajectoryLen=50)
        
        # Analyze action distribution
        action_distribution = self._calculate_action_distribution(agent, mdp)
        
        # Calculate policy entropy
        policy_entropy = self._calculate_policy_entropy(agent, mdp)
        
        # Equivalence classes
        equiv_classes = self._analyze_equivalence_classes(agent, mdp, "Default")
        
        self.results['Default_Parking'] = {
            'agent': agent,
            'rewards': rewards,
            'mdp': mdp,
            'start': start,
            'action_distribution': action_distribution,
            'policy_entropy': policy_entropy,
            'equivalence_classes': equiv_classes,
            'final_reward': np.mean(rewards[-10:]),
            'num_epochs': num_epochs
        }
        
        print(f"\nDefault NN Training Complete. Final Average Reward: {np.mean(rewards[-10:]):.2f}")
        
        return agent, rewards, mdp
    
    def _calculate_action_distribution(self, agent, mdp):
        """Calculate action distribution across all states"""
        action_counts = defaultdict(int)
        state_action_pairs = []
        
        for state in range(mdp.numStates):
            q_values = agent.predict_qvalues(state)
            best_action = np.argmax(q_values)
            action_counts[best_action] += 1
            state_action_pairs.append((state, best_action))
        
        total = sum(action_counts.values())
        action_probs = {action: count/total for action, count in action_counts.items()}
        
        # Calculate uniformity (chi-square test for uniform distribution)
        expected_count = total / mdp.numActions
        chi_square_stat = sum((count - expected_count)**2 / expected_count 
                             for count in action_counts.values())
        p_value = 1 - chi2_contingency([list(action_counts.values())])[1] if total > 0 else 0
        
        return {
            'action_counts': dict(action_counts),
            'action_probs': action_probs,
            'total_states': total,
            'uniformity_score': 1 / (1 + chi_square_stat / (mdp.numActions - 1)),  # Normalized score
            'chi_square_stat': chi_square_stat,
            'p_value': p_value,
            'is_uniform': p_value > 0.05 if p_value else False
        }
    
    def _calculate_policy_entropy(self, agent, mdp, temperature=1.0):
        """Calculate policy entropy for each state"""
        entropies = []
        
        for state in range(mdp.numStates):
            q_values = agent.predict_qvalues(state)
            
            # Convert to probabilities using softmax
            exp_q = np.exp(q_values / temperature - np.max(q_values / temperature))
            probs = exp_q / np.sum(exp_q)
            
            # Calculate entropy
            state_entropy = scipy_entropy(probs)
            entropies.append(state_entropy)
        
        return {
            'state_entropies': np.array(entropies),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'median_entropy': np.median(entropies)
        }
    
    def _analyze_equivalence_classes(self, agent, mdp, network_name):
        """Analyze equivalence classes between parking states"""
        state_qvalues = []
        state_info = []
        
        for state in range(mdp.numStates):
            q_vals = agent.predict_qvalues(state)
            state_qvalues.append(q_vals)
            
            # Try to get state type
            try:
                state_type = StateType.get(state)
                state_type_name = state_type.name if hasattr(state_type, 'name') else str(state_type)
            except:
                state_type_name = 'UNKNOWN'
            
            state_info.append({
                'state': state,
                'type': state_type_name,
                'q_values': q_vals,
                'best_action': np.argmax(q_vals),
                'q_variance': np.var(q_vals),
                'q_spread': np.max(q_vals) - np.min(q_vals),
                'q_max': np.max(q_vals)
            })
        
        state_qvalues = np.array(state_qvalues)
        
        # Cluster based on Q-values
        n_clusters = min(6, max(2, mdp.numStates // 4))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed_value, n_init=10)
            clusters = kmeans.fit_predict(state_qvalues)
            kmeans_model = kmeans
        except (AttributeError, Exception) as e:
            # Fallback to simpler clustering if KMeans fails (e.g., threadpoolctl issues on macOS)
            print(f"Warning: KMeans clustering failed ({e}), using simple distance-based clustering")
            # Use simple distance-based clustering as fallback
            try:
                linkage_matrix = linkage(state_qvalues, method='ward')
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                kmeans_model = None
            except:
                # Ultimate fallback: cluster by best action
                clusters = np.array([s['best_action'] for s in state_info])
                kmeans_model = None
        
        # PCA for visualization
        if mdp.numActions > 1:
            pca = PCA(n_components=min(2, mdp.numActions))
            state_2d = pca.fit_transform(state_qvalues)
        else:
            state_2d = state_qvalues
        
        # Group states by cluster
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(state_info[i])
        
        return {
            'clusters': clusters,
            'cluster_groups': dict(cluster_groups),
            'num_clusters': len(set(clusters)),
            'state_2d': state_2d,
            'state_info': state_info,
            'state_qvalues': state_qvalues,
            'kmeans_model': kmeans_model
        }
    
    def generate_individual_metric_visualizations(self):
        """Generate separate visualization figures for each metric"""
        print("\nGenerating visualizations...")
        
        # 1. Learning Curves
        self._plot_learning_curves()
        
        # 2. Equivalence Classes
        self._plot_equivalence_classes()
        
        # 3. Entropy Analysis
        self._plot_entropy_analysis()
        
        # 4. Action Distribution Uniformity
        self._plot_action_distribution()
        
        # 5. Qualitative Analysis
        self._plot_qualitative_analysis()
    
    def _plot_learning_curves(self):
        """Generate detailed learning curve visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
        
        # CNN Learning Curve
        if 'CNN_CIFAR10' in self.results:
            ax = axes[0, 0]
            cnn_data = self.results['CNN_CIFAR10']
            epochs = range(1, len(cnn_data['test_accuracies']) + 1)
            
            ax.plot(epochs, cnn_data['train_accuracies'], 'b-', label='Train Accuracy', 
                   linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, cnn_data['test_accuracies'], 'r-', label='Test Accuracy', 
                   linewidth=2, marker='s', markersize=4)
            ax.plot(epochs, cnn_data['train_losses'], 'g--', label='Train Loss', 
                   linewidth=1.5, alpha=0.7)
            ax.plot(epochs, cnn_data['test_losses'], 'm--', label='Test Loss', 
                   linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%) / Loss', fontsize=12)
            ax.set_title('CNN on CIFAR-10', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add final performance annotation
            final_acc = cnn_data['test_accuracies'][-1]
            ax.annotate(f'Final: {final_acc:.2f}%', 
                       xy=(len(epochs), final_acc),
                       xytext=(len(epochs)*0.7, final_acc*0.8),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, fontweight='bold')
        
        # LSTM Learning Curve
        if 'LSTM_Parking' in self.results:
            ax = axes[0, 1]
            lstm_data = self.results['LSTM_Parking']
            epochs = range(1, len(lstm_data['rewards']) + 1)
            rewards = lstm_data['rewards']
            
            ax.plot(epochs, rewards, 'g-', label='Average Reward', 
                   linewidth=2, marker='o', markersize=4)
            
            # Add moving average
            window = max(5, len(rewards) // 10)
            if len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(epochs[window-1:], moving_avg, 'b--', 
                       label=f'Moving Avg (window={window})', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Average Reward', fontsize=12)
            ax.set_title('LSTM on Parking MDP', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add improvement annotation
            improvement = rewards[-1] - rewards[0]
            ax.annotate(f'Improvement: {improvement:.2f}', 
                       xy=(len(epochs), rewards[-1]),
                       xytext=(len(epochs)*0.7, rewards[-1]*0.8),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, fontweight='bold')
        
        # Default NN Learning Curve
        if 'Default_Parking' in self.results:
            ax = axes[1, 0]
            default_data = self.results['Default_Parking']
            epochs = range(1, len(default_data['rewards']) + 1)
            rewards = default_data['rewards']
            
            ax.plot(epochs, rewards, 'r-', label='Average Reward', 
                   linewidth=2, marker='s', markersize=4)
            
            # Add moving average
            window = max(5, len(rewards) // 10)
            if len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(epochs[window-1:], moving_avg, 'b--', 
                       label=f'Moving Avg (window={window})', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Average Reward', fontsize=12)
            ax.set_title('Default NN on Parking MDP', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add improvement annotation
            improvement = rewards[-1] - rewards[0]
            ax.annotate(f'Improvement: {improvement:.2f}', 
                       xy=(len(epochs), rewards[-1]),
                       xytext=(len(epochs)*0.7, rewards[-1]*0.8),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, fontweight='bold')
        
        # Combined Learning Curves Comparison
        ax = axes[1, 1]
        if 'LSTM_Parking' in self.results and 'Default_Parking' in self.results:
            lstm_data = self.results['LSTM_Parking']
            default_data = self.results['Default_Parking']
            
            epochs_lstm = range(1, len(lstm_data['rewards']) + 1)
            epochs_default = range(1, len(default_data['rewards']) + 1)
            
            ax.plot(epochs_lstm, lstm_data['rewards'], 'g-', 
                   label='LSTM', linewidth=2, marker='o', markersize=3)
            ax.plot(epochs_default, default_data['rewards'], 'r-', 
                   label='Default NN', linewidth=2, marker='s', markersize=3)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Average Reward', fontsize=12)
            ax.set_title('Learning Curves Comparison (MDP Networks)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_curves_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_equivalence_classes(self):
        """Generate equivalence classes visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Equivalence Classes Between Parking States', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                    ('Default_Parking', 'Default', 'red')]:
            if name in self.results:
                equiv_data = self.results[name]['equivalence_classes']
                state_2d = equiv_data['state_2d']
                clusters = equiv_data['clusters']
                
                if state_2d.shape[1] >= 2:
                    # PCA Scatter Plot
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    scatter = ax.scatter(state_2d[:, 0], state_2d[:, 1], 
                                       c=clusters, cmap='tab10', 
                                       alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
                    ax.set_xlabel('PCA Component 1', fontsize=12)
                    ax.set_ylabel('PCA Component 2', fontsize=12)
                    ax.set_title(f'{label} Network - State Equivalence Classes\n'
                               f'{equiv_data["num_clusters"]} clusters discovered', 
                               fontsize=14, fontweight='bold')
                    plt.colorbar(scatter, ax=ax, label='Cluster ID')
                    ax.grid(True, alpha=0.3)
                    
                    # Add cluster centers
                    if equiv_data['kmeans_model'] is not None and hasattr(equiv_data['kmeans_model'], 'cluster_centers_'):
                        centers_2d = equiv_data['kmeans_model'].cluster_centers_
                        if centers_2d.shape[1] >= 2:
                            # Project centers to 2D if needed
                            pca = PCA(n_components=2)
                            if equiv_data['state_qvalues'].shape[1] > 2:
                                centers_2d = pca.fit_transform(equiv_data['state_qvalues'])[:equiv_data['num_clusters']]
                            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                                     c='red', marker='X', s=200, 
                                     label='Cluster Centers', edgecolors='black', linewidth=2)
                            ax.legend()
                    
                    plot_idx += 1
                    
                    # Cluster Size Distribution
                    if plot_idx <= 3:
                        ax = axes[plot_idx // 2, plot_idx % 2]
                        cluster_sizes = [len(states) for states in equiv_data['cluster_groups'].values()]
                        cluster_ids = sorted(equiv_data['cluster_groups'].keys())
                        
                        bars = ax.bar(cluster_ids, cluster_sizes, color=color, alpha=0.7)
                        ax.set_xlabel('Cluster ID', fontsize=12)
                        ax.set_ylabel('Number of States', fontsize=12)
                        ax.set_title(f'{label} Network - Cluster Size Distribution', 
                                   fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels
                        for bar, size in zip(bars, cluster_sizes):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'{size}', ha='center', va='bottom', fontweight='bold')
                        
                        plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('equivalence_classes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_entropy_analysis(self):
        """Generate entropy analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Policy Entropy Analysis', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                    ('Default_Parking', 'Default', 'red')]:
            if name in self.results:
                ent_data = self.results[name]['policy_entropy']
                entropies = ent_data['state_entropies']
                
                # Histogram
                ax = axes[plot_idx // 2, plot_idx % 2]
                n, bins, patches = ax.hist(entropies, bins=20, color=color, alpha=0.7, 
                                          edgecolor='black', linewidth=1)
                ax.axvline(ent_data['mean_entropy'], color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {ent_data['mean_entropy']:.3f}")
                ax.axvline(ent_data['median_entropy'], color='blue', linestyle='--', 
                          linewidth=2, label=f"Median: {ent_data['median_entropy']:.3f}")
                ax.set_xlabel('Policy Entropy', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{label} Network - Entropy Distribution', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                plot_idx += 1
                
                # Box Plot Comparison
                if plot_idx == 2:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    entropies_list = []
                    labels_list = []
                    colors_list = []
                    
                    for n2, l2, c2 in [('LSTM_Parking', 'LSTM', 'green'), 
                                       ('Default_Parking', 'Default', 'red')]:
                        if n2 in self.results:
                            entropies_list.append(self.results[n2]['policy_entropy']['state_entropies'])
                            labels_list.append(l2)
                            colors_list.append(c2)
                    
                    bp = ax.boxplot(entropies_list, labels=labels_list, patch_artist=True)
                    for patch, color in zip(bp['boxes'], colors_list):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel('Policy Entropy', fontsize=12)
                    ax.set_title('Entropy Comparison Across Networks', 
                               fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plot_idx += 1
                
                # State-by-State Entropy
                if plot_idx <= 3:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    states = range(len(entropies))
                    ax.plot(states, entropies, color=color, linewidth=1.5, alpha=0.7)
                    ax.fill_between(states, entropies, alpha=0.3, color=color)
                    ax.axhline(ent_data['mean_entropy'], color='red', linestyle='--', 
                              linewidth=2, label=f"Mean: {ent_data['mean_entropy']:.3f}")
                    ax.set_xlabel('State Index', fontsize=12)
                    ax.set_ylabel('Policy Entropy', fontsize=12)
                    ax.set_title(f'{label} Network - Entropy by State', 
                               fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('entropy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_distribution(self):
        """Generate action distribution uniformity visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Action Distribution Uniformity Analysis', fontsize=16, fontweight='bold')
        
        # CNN Class Distribution
        if 'CNN_CIFAR10' in self.results:
            ax = axes[0, 0]
            cnn_data = self.results['CNN_CIFAR10']
            class_dist = cnn_data['class_distribution']
            
            classes = sorted(class_dist.keys())
            counts = [class_dist[c] for c in classes]
            class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
            
            bars = ax.bar(classes, counts, color='blue', alpha=0.7, edgecolor='black')
            
            # Add uniform line
            if cnn_data['total_samples'] > 0:
                expected = cnn_data['total_samples'] / len(classes)
                ax.axhline(expected, color='red', linestyle='--', 
                          linewidth=2, label=f'Uniform: {expected:.1f}')
            
            ax.set_xlabel('CIFAR-10 Class', fontsize=12)
            ax.set_ylabel('Number of Predictions', fontsize=12)
            ax.set_title('CNN - Class Prediction Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(classes)
            ax.set_xticklabels([class_names[c] for c in classes], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom', fontsize=8)
        
        # MDP Networks Action Distribution
        plot_idx = 1
        for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                    ('Default_Parking', 'Default', 'red')]:
            if name in self.results and plot_idx < 4:
                action_dist = self.results[name]['action_distribution']
                ax = axes[plot_idx // 2, plot_idx % 2]
                
                actions = sorted(action_dist['action_counts'].keys())
                counts = [action_dist['action_counts'][a] for a in actions]
                
                bars = ax.bar(actions, counts, color=color, alpha=0.7, edgecolor='black')
                
                # Add uniform line
                if action_dist['total_states'] > 0:
                    expected = action_dist['total_states'] / len(actions)
                    ax.axhline(expected, color='black', linestyle='--', 
                             linewidth=2, label=f'Uniform: {expected:.1f}')
                
                # Add uniformity score
                uniformity = action_dist['uniformity_score']
                ax.text(0.02, 0.98, f'Uniformity Score: {uniformity:.3f}\n'
                       f'χ² = {action_dist["chi_square_stat"]:.2f}\n'
                       f'p = {action_dist["p_value"]:.4f}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
                
                ax.set_xlabel('Action', fontsize=12)
                ax.set_ylabel('Number of States', fontsize=12)
                ax.set_title(f'{label} Network - Action Distribution', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontweight='bold')
                
                plot_idx += 1
        
        # Uniformity Comparison
        if plot_idx <= 3:
            ax = axes[1, 1]
            networks = []
            uniformity_scores = []
            colors_bar = []
            
            if 'CNN_CIFAR10' in self.results:
                cnn_data = self.results['CNN_CIFAR10']
                class_dist = cnn_data['class_distribution']
                total = cnn_data['total_samples']
                expected = total / 10 if total > 0 else 0
                chi_sq = sum((count - expected)**2 / expected for count in class_dist.values()) if expected > 0 else 0
                uniformity = 1 / (1 + chi_sq / 9) if chi_sq > 0 else 0
                networks.append('CNN')
                uniformity_scores.append(uniformity)
                colors_bar.append('blue')
            
            for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                       ('Default_Parking', 'Default', 'red')]:
                if name in self.results:
                    networks.append(label)
                    uniformity_scores.append(self.results[name]['action_distribution']['uniformity_score'])
                    colors_bar.append(color)
            
            bars = ax.bar(networks, uniformity_scores, color=colors_bar, alpha=0.7, edgecolor='black')
            ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
                      label='Perfect Uniformity (1.0)')
            ax.set_ylabel('Uniformity Score', fontsize=12)
            ax.set_title('Uniformity Comparison Across Networks', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, score in zip(bars, uniformity_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('action_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_qualitative_analysis(self):
        """Generate qualitative analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Qualitative Analysis - Q-Value Patterns', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                    ('Default_Parking', 'Default', 'red')]:
            if name in self.results and plot_idx < 4:
                equiv_data = self.results[name]['equivalence_classes']
                state_info = equiv_data['state_info']
                
                # Q-Value Variance Distribution
                ax = axes[plot_idx // 2, plot_idx % 2]
                q_variances = [s['q_variance'] for s in state_info]
                ax.hist(q_variances, bins=20, color=color, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(q_variances), color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {np.mean(q_variances):.4f}")
                ax.set_xlabel('Q-Value Variance', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{label} Network - Q-Value Variance', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                plot_idx += 1
                
                # Q-Value Spread Distribution
                if plot_idx < 4:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                    q_spreads = [s['q_spread'] for s in state_info]
                    ax.hist(q_spreads, bins=20, color=color, alpha=0.7, edgecolor='black')
                    ax.axvline(np.mean(q_spreads), color='red', linestyle='--', 
                              linewidth=2, label=f"Mean: {np.mean(q_spreads):.4f}")
                    ax.set_xlabel('Q-Value Spread (Max - Min)', fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'{label} Network - Q-Value Spread', 
                               fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plot_idx += 1
        
        # Q-Value Comparison Heatmap (if space available)
        if plot_idx < 4:
            ax = axes[1, 1]
            # Compare variance and spread across networks
            networks = []
            variances = []
            spreads = []
            colors_list = []
            
            for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                       ('Default_Parking', 'Default', 'red')]:
                if name in self.results:
                    equiv_data = self.results[name]['equivalence_classes']
                    state_info = equiv_data['state_info']
                    networks.append(label)
                    variances.append(np.mean([s['q_variance'] for s in state_info]))
                    spreads.append(np.mean([s['q_spread'] for s in state_info]))
                    colors_list.append(color)
            
            x_pos = np.arange(len(networks))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, variances, width, label='Mean Variance', 
                          color=colors_list, alpha=0.7, edgecolor='black')
            ax2 = ax.twinx()
            bars2 = ax2.bar(x_pos + width/2, spreads, width, label='Mean Spread', 
                           color=colors_list, alpha=0.5, edgecolor='black')
            
            ax.set_xlabel('Network', fontsize=12)
            ax.set_ylabel('Q-Value Variance', fontsize=12, color='black')
            ax2.set_ylabel('Q-Value Spread', fontsize=12, color='gray')
            ax.set_title('Q-Value Pattern Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(networks)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('qualitative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        
        # Generate individual metric visualizations
        self.generate_individual_metric_visualizations()
        
        # Also generate combined overview
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Learning Curves (Top row)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'CNN_CIFAR10' in self.results:
            cnn_data = self.results['CNN_CIFAR10']
            epochs = range(1, len(cnn_data['test_accuracies']) + 1)
            ax1.plot(epochs, cnn_data['train_accuracies'], 'b-', label='CNN Train Acc', linewidth=2)
            ax1.plot(epochs, cnn_data['test_accuracies'], 'b--', label='CNN Test Acc', linewidth=2)
        
        if 'LSTM_Parking' in self.results:
            lstm_data = self.results['LSTM_Parking']
            epochs = range(1, len(lstm_data['rewards']) + 1)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(epochs, lstm_data['rewards'], 'g-', label='LSTM Rewards', linewidth=2, alpha=0.7)
            ax1_twin.set_ylabel('Average Reward', color='g')
        
        if 'Default_Parking' in self.results:
            default_data = self.results['Default_Parking']
            epochs = range(1, len(default_data['rewards']) + 1)
            if 'ax1_twin' not in locals():
                ax1_twin = ax1.twinx()
            ax1_twin.plot(epochs, default_data['rewards'], 'r-', label='Default Rewards', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)', color='b')
        ax1.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        if 'ax1_twin' in locals():
            ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Entropy Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        entropies_to_plot = []
        labels = []
        colors = []
        
        if 'LSTM_Parking' in self.results:
            lstm_entropy = self.results['LSTM_Parking']['policy_entropy']['state_entropies']
            entropies_to_plot.append(lstm_entropy)
            labels.append('LSTM')
            colors.append('green')
        
        if 'Default_Parking' in self.results:
            default_entropy = self.results['Default_Parking']['policy_entropy']['state_entropies']
            entropies_to_plot.append(default_entropy)
            labels.append('Default')
            colors.append('red')
        
        if entropies_to_plot:
            ax2.hist(entropies_to_plot, bins=15, alpha=0.6, label=labels, color=colors[:len(labels)], density=True)
            ax2.set_xlabel('Policy Entropy')
            ax2.set_ylabel('Density')
            ax2.set_title('Policy Entropy Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3-4. Equivalence Classes (LSTM and Default)
        plot_idx = 1
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                equiv_data = self.results[name]['equivalence_classes']
                state_2d = equiv_data['state_2d']
                clusters = equiv_data['clusters']
                
                if state_2d.shape[1] >= 2:
                    ax = fig.add_subplot(gs[1, plot_idx-1])
                    scatter = ax.scatter(state_2d[:, 0], state_2d[:, 1], c=clusters, 
                                       cmap='tab10', alpha=0.7, s=100)
                    ax.set_title(f'{label} Equivalence Classes\n({equiv_data["num_clusters"]} classes)', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    plt.colorbar(scatter, ax=ax)
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1
        
        # 5-6. Action Distribution Uniformity
        plot_idx = 0
        for name, label, color in [('LSTM_Parking', 'LSTM', 'green'), 
                                    ('Default_Parking', 'Default', 'red')]:
            if name in self.results:
                action_dist = self.results[name]['action_distribution']
                ax = fig.add_subplot(gs[2, plot_idx])
                
                actions = sorted(action_dist['action_counts'].keys())
                counts = [action_dist['action_counts'][a] for a in actions]
                
                bars = ax.bar(actions, counts, color=color, alpha=0.7)
                ax.set_xlabel('Action')
                ax.set_ylabel('Number of States')
                ax.set_title(f'{label} Action Distribution\n'
                           f'Uniformity: {action_dist["uniformity_score"]:.3f}',
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add expected uniform line
                if action_dist['total_states'] > 0:
                    expected = action_dist['total_states'] / len(actions)
                    ax.axhline(expected, color='black', linestyle='--', 
                             linewidth=2, label=f'Uniform: {expected:.1f}')
                    ax.legend()
                
                plot_idx += 1
        
        # 7. CNN Class Distribution
        if 'CNN_CIFAR10' in self.results:
            ax7 = fig.add_subplot(gs[2, 2:])
            cnn_data = self.results['CNN_CIFAR10']
            class_dist = cnn_data['class_distribution']
            
            classes = sorted(class_dist.keys())
            counts = [class_dist[c] for c in classes]
            
            bars = ax7.bar(classes, counts, color='blue', alpha=0.7)
            ax7.set_xlabel('CIFAR-10 Class')
            ax7.set_ylabel('Number of Predictions')
            ax7.set_title('CNN Class Prediction Distribution', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='y')
            
            # Add uniform line
            if cnn_data['total_samples'] > 0:
                expected = cnn_data['total_samples'] / len(classes)
                ax7.axhline(expected, color='black', linestyle='--', 
                          linewidth=2, label=f'Uniform: {expected:.1f}')
                ax7.legend()
        
        # 8-9. Final Performance Summary
        ax8 = fig.add_subplot(gs[3, :2])
        networks = []
        metrics = []
        colors_bar = []
        
        if 'CNN_CIFAR10' in self.results:
            networks.append('CNN\n(CIFAR-10)')
            metrics.append(self.results['CNN_CIFAR10']['final_test_acc'])
            colors_bar.append('blue')
        
        if 'LSTM_Parking' in self.results:
            networks.append('LSTM\n(Parking)')
            metrics.append(self.results['LSTM_Parking']['final_reward'])
            colors_bar.append('green')
        
        if 'Default_Parking' in self.results:
            networks.append('Default\n(Parking)')
            metrics.append(self.results['Default_Parking']['final_reward'])
            colors_bar.append('red')
        
        if networks:
            bars = ax8.bar(networks, metrics, color=colors_bar, alpha=0.7)
            ax8.set_ylabel('Final Performance')
            ax8.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, metric in zip(bars, metrics):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{metric:.2f}',
                        ha='center', va='bottom', fontweight='bold')
        
        # 10. Entropy Statistics Table
        ax9 = fig.add_subplot(gs[3, 2:])
        ax9.axis('off')
        
        table_data = []
        headers = ['Network', 'Mean Entropy', 'Std Entropy', 'Min', 'Max']
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                ent_data = self.results[name]['policy_entropy']
                table_data.append([
                    label,
                    f"{ent_data['mean_entropy']:.3f}",
                    f"{ent_data['std_entropy']:.3f}",
                    f"{ent_data['min_entropy']:.3f}",
                    f"{ent_data['max_entropy']:.3f}"
                ])
        
        if table_data:
            table = ax9.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax9.set_title('Policy Entropy Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Comprehensive Neural Network Analysis Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('comprehensive_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def generate_quantitative_report(self):
        """Generate comprehensive quantitative report"""
        print("\nGenerating quantitative report...")
        
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("NEURAL NETWORK ARCHITECTURES: QUANTITATIVE ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. Methodology Section
        report_lines.append("="*80)
        report_lines.append("1. METHODOLOGY")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("This report analyzes three distinct neural network architectures:")
        report_lines.append("")
        
        # CNN Methodology
        if 'CNN_CIFAR10' in self.results:
            cnn_data = self.results['CNN_CIFAR10']
            report_lines.append("1.1 CNN (Convolutional Neural Network)")
            report_lines.append("-" * 40)
            report_lines.append(f"   • Architecture: Multi-layer CNN with conv channels [32, 64, 128]")
            report_lines.append(f"   • Dataset: CIFAR-10 (32x32 RGB images, 10 classes)")
            report_lines.append(f"   • Training epochs: {cnn_data['num_epochs']}")
            report_lines.append(f"   • Final test accuracy: {cnn_data['final_test_acc']:.2f}%")
            report_lines.append(f"   • Purpose: Image classification task")
            report_lines.append("")
        
        # LSTM Methodology
        if 'LSTM_Parking' in self.results:
            lstm_data = self.results['LSTM_Parking']
            report_lines.append("1.2 LSTM (Long Short-Term Memory)")
            report_lines.append("-" * 40)
            report_lines.append(f"   • Architecture: LSTM with hidden_size=64, num_layers=2")
            report_lines.append(f"   • Dataset: Parking MDP ({lstm_data['mdp'].numStates} states, {lstm_data['mdp'].numActions} actions)")
            report_lines.append(f"   • Training epochs: {lstm_data['num_epochs']}")
            report_lines.append(f"   • Final average reward: {lstm_data['final_reward']:.2f}")
            report_lines.append(f"   • Purpose: Sequential decision-making in reinforcement learning")
            report_lines.append("")
        
        # Default Methodology
        if 'Default_Parking' in self.results:
            default_data = self.results['Default_Parking']
            report_lines.append("1.3 Default/Standard Neural Network")
            report_lines.append("-" * 40)
            report_lines.append(f"   • Architecture: Fully connected layers [128, 64]")
            report_lines.append(f"   • Dataset: Parking MDP ({default_data['mdp'].numStates} states, {default_data['mdp'].numActions} actions)")
            report_lines.append(f"   • Training epochs: {default_data['num_epochs']}")
            report_lines.append(f"   • Final average reward: {default_data['final_reward']:.2f}")
            report_lines.append(f"   • Purpose: Baseline reinforcement learning agent")
            report_lines.append("")
        
        # 2. Equivalence Classes
        report_lines.append("="*80)
        report_lines.append("2. EQUIVALENCE CLASSES BETWEEN PARKING STATES")
        report_lines.append("="*80)
        report_lines.append("")
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                equiv_data = self.results[name]['equivalence_classes']
                report_lines.append(f"2.{1 if name == 'LSTM_Parking' else 2} {label} Network")
                report_lines.append("-" * 40)
                report_lines.append(f"   • Number of equivalence classes discovered: {equiv_data['num_clusters']}")
                report_lines.append(f"   • Clustering method: K-means on Q-value vectors")
                report_lines.append("")
                
                # Cluster details
                for cluster_id, states in equiv_data['cluster_groups'].items():
                    report_lines.append(f"   Cluster {cluster_id}: {len(states)} states")
                    # Show sample states from cluster
                    sample_states = [s['state'] for s in states[:5]]
                    report_lines.append(f"     Sample states: {sample_states}")
                report_lines.append("")
        
        # 3. Qualitative Analysis
        report_lines.append("="*80)
        report_lines.append("3. QUALITATIVE ANALYSIS")
        report_lines.append("="*80)
        report_lines.append("")
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                data = self.results[name]
                equiv_data = data['equivalence_classes']
                report_lines.append(f"3.{1 if name == 'LSTM_Parking' else 2} {label} Network")
                report_lines.append("-" * 40)
                
                # Q-value statistics
                q_variances = [s['q_variance'] for s in equiv_data['state_info']]
                q_spreads = [s['q_spread'] for s in equiv_data['state_info']]
                
                report_lines.append(f"   Q-Value Analysis:")
                report_lines.append(f"   • Mean Q-value variance: {np.mean(q_variances):.4f}")
                report_lines.append(f"   • Mean Q-value spread: {np.mean(q_spreads):.4f}")
                report_lines.append(f"   • Policy determinism: High spread indicates clearer preferences")
                
                # Action diversity
                action_dist = data['action_distribution']
                report_lines.append(f"   Action Diversity:")
                report_lines.append(f"   • Actions utilized: {len(action_dist['action_counts'])}/{data['mdp'].numActions}")
                report_lines.append(f"   • Most common action: {max(action_dist['action_counts'].items(), key=lambda x: x[1])[0]}")
                report_lines.append("")
        
        # 4. Entropy
        report_lines.append("="*80)
        report_lines.append("4. ENTROPY ANALYSIS")
        report_lines.append("="*80)
        report_lines.append("")
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                ent_data = self.results[name]['policy_entropy']
                report_lines.append(f"4.{1 if name == 'LSTM_Parking' else 2} {label} Network")
                report_lines.append("-" * 40)
                report_lines.append(f"   • Mean entropy: {ent_data['mean_entropy']:.4f}")
                report_lines.append(f"   • Std entropy: {ent_data['std_entropy']:.4f}")
                report_lines.append(f"   • Min entropy: {ent_data['min_entropy']:.4f}")
                report_lines.append(f"   • Max entropy: {ent_data['max_entropy']:.4f}")
                report_lines.append(f"   • Median entropy: {ent_data['median_entropy']:.4f}")
                
                # Interpretation
                if ent_data['mean_entropy'] > np.log(self.results[name]['mdp'].numActions) * 0.7:
                    interpretation = "High entropy - Policy is more exploratory/random"
                elif ent_data['mean_entropy'] < np.log(self.results[name]['mdp'].numActions) * 0.3:
                    interpretation = "Low entropy - Policy is more deterministic"
                else:
                    interpretation = "Moderate entropy - Balanced exploration/exploitation"
                
                report_lines.append(f"   • Interpretation: {interpretation}")
                report_lines.append("")
        
        # 5. Learning Curves
        report_lines.append("="*80)
        report_lines.append("5. LEARNING CURVES")
        report_lines.append("="*80)
        report_lines.append("")
        
        if 'CNN_CIFAR10' in self.results:
            cnn_data = self.results['CNN_CIFAR10']
            report_lines.append("5.1 CNN on CIFAR-10")
            report_lines.append("-" * 40)
            report_lines.append(f"   • Initial test accuracy: {cnn_data['test_accuracies'][0]:.2f}%")
            report_lines.append(f"   • Final test accuracy: {cnn_data['test_accuracies'][-1]:.2f}%")
            report_lines.append(f"   • Improvement: {cnn_data['test_accuracies'][-1] - cnn_data['test_accuracies'][0]:.2f}%")
            report_lines.append(f"   • Learning trend: {'Increasing' if cnn_data['test_accuracies'][-1] > cnn_data['test_accuracies'][0] else 'Stable/Decreasing'}")
            report_lines.append("")
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                data = self.results[name]
                rewards = data['rewards']
                report_lines.append(f"5.{2 if name == 'LSTM_Parking' else 3} {label} on Parking MDP")
                report_lines.append("-" * 40)
                report_lines.append(f"   • Initial average reward: {rewards[0]:.2f}")
                report_lines.append(f"   • Final average reward: {rewards[-1]:.2f}")
                report_lines.append(f"   • Best average reward: {max(rewards):.2f}")
                report_lines.append(f"   • Improvement: {rewards[-1] - rewards[0]:.2f}")
                report_lines.append(f"   • Learning stability (std of last 10): {np.std(rewards[-10:]):.4f}")
                report_lines.append("")
        
        # 6. Uniformness of Action Distribution
        report_lines.append("="*80)
        report_lines.append("6. UNIFORMNESS OF ACTION DISTRIBUTION")
        report_lines.append("="*80)
        report_lines.append("")
        
        if 'CNN_CIFAR10' in self.results:
            cnn_data = self.results['CNN_CIFAR10']
            class_dist = cnn_data['class_distribution']
            total = cnn_data['total_samples']
            expected = total / 10 if total > 0 else 0
            
            chi_sq = sum((count - expected)**2 / expected for count in class_dist.values()) if expected > 0 else 0
            uniformity = 1 / (1 + chi_sq / 9) if chi_sq > 0 else 0
            
            report_lines.append("6.1 CNN on CIFAR-10 (Class Distribution)")
            report_lines.append("-" * 40)
            report_lines.append(f"   • Uniformity score: {uniformity:.4f} (1.0 = perfectly uniform)")
            report_lines.append(f"   • Chi-square statistic: {chi_sq:.4f}")
            report_lines.append(f"   • Expected uniform count per class: {expected:.1f}")
            report_lines.append(f"   • Most predicted class: {max(class_dist.items(), key=lambda x: x[1])[0]}")
            report_lines.append(f"   • Least predicted class: {min(class_dist.items(), key=lambda x: x[1])[0]}")
            report_lines.append("")
        
        for name, label in [('LSTM_Parking', 'LSTM'), ('Default_Parking', 'Default')]:
            if name in self.results:
                action_dist = self.results[name]['action_distribution']
                report_lines.append(f"6.{2 if name == 'LSTM_Parking' else 3} {label} on Parking MDP")
                report_lines.append("-" * 40)
                report_lines.append(f"   • Uniformity score: {action_dist['uniformity_score']:.4f} (1.0 = perfectly uniform)")
                report_lines.append(f"   • Chi-square statistic: {action_dist['chi_square_stat']:.4f}")
                report_lines.append(f"   • P-value: {action_dist['p_value']:.4f}")
                report_lines.append(f"   • Distribution uniform: {'Yes' if action_dist['is_uniform'] else 'No'} (p > 0.05)")
                report_lines.append("")
                
                # Action breakdown
                for action, count in sorted(action_dist['action_counts'].items()):
                    percentage = (count / action_dist['total_states']) * 100
                    report_lines.append(f"     Action {action}: {count} states ({percentage:.1f}%)")
                report_lines.append("")
        
        # Summary
        report_lines.append("="*80)
        report_lines.append("SUMMARY")
        report_lines.append("="*80)
        report_lines.append("")
        
        if 'CNN_CIFAR10' in self.results:
            report_lines.append("• CNN successfully learned CIFAR-10 classification task")
        if 'LSTM_Parking' in self.results:
            report_lines.append(f"• LSTM achieved {self.results['LSTM_Parking']['final_reward']:.2f} average reward on Parking MDP")
        if 'Default_Parking' in self.results:
            report_lines.append(f"• Default NN achieved {self.results['Default_Parking']['final_reward']:.2f} average reward on Parking MDP")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open('quantitative_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nReport saved to 'quantitative_analysis_report.txt'")
        
        return report_text
    
    def load_existing_results(self, filename='analysis_results.pkl'):
        """Load existing results from pickle file"""
        try:
            import os
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    loaded_results = pickle.load(f)
                    print(f"Loaded existing results from {filename}")
                    return loaded_results
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
        return None
    
    def run_complete_analysis(self, cnn_epochs=20, mdp_epochs=50, skip_training=False, load_existing=True, skip_cnn=False):
        """Run complete analysis pipeline"""
        print("\nStarting comprehensive neural network analysis...")
        
        # Try to load existing results first
        if load_existing:
            existing = self.load_existing_results()
            if existing:
                self.results = existing
                print(f"\nUsing existing training results for: {list(existing.keys())}\n")
        
        # Train networks (only if not skipping)
        if not skip_training:
            # CNN training
            if 'CNN_CIFAR10' not in self.results:
                if skip_cnn:
                    print("Skipping CNN training (--skip-cnn flag).")
                else:
                    print("\nTraining CNN on CIFAR-10...")
                    self.train_cnn_on_cifar10(num_epochs=cnn_epochs)
            
            # LSTM training
            if 'LSTM_Parking' not in self.results:
                print("\nTraining LSTM on Parking MDP...")
                self.train_lstm_on_mdp(num_epochs=mdp_epochs)
            
            # Default NN training
            if 'Default_Parking' not in self.results:
                print("\nTraining Default NN on Parking MDP...")
                self.train_default_on_mdp(num_epochs=mdp_epochs)
        else:
            # Even if loading, we need to ensure all analysis metrics are computed
            for name in ['LSTM_Parking', 'Default_Parking']:
                if name in self.results:
                    agent = self.results[name]['agent']
                    mdp = self.results[name]['mdp']
                    
                    # Recompute metrics if missing
                    if 'action_distribution' not in self.results[name]:
                        self.results[name]['action_distribution'] = self._calculate_action_distribution(agent, mdp)
                    
                    if 'policy_entropy' not in self.results[name]:
                        self.results[name]['policy_entropy'] = self._calculate_policy_entropy(agent, mdp)
                    
                    if 'equivalence_classes' not in self.results[name]:
                        self.results[name]['equivalence_classes'] = self._analyze_equivalence_classes(agent, mdp, name)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        self.generate_quantitative_report()
        
        # Save results
        with open('analysis_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print("\nAnalysis complete.")
        
        return self.results


def main():
    """Main execution function"""
    import sys
    
    analyzer = ComprehensiveNetworkAnalyzer(seed_value=42)
    
    # Check command line arguments
    skip_cnn = '--skip-cnn' in sys.argv or '--no-cnn' in sys.argv
    train_default_only = '--default-only' in sys.argv
    
    if skip_cnn or train_default_only:
        print("\nSkipping CNN training.\n")
    
    # Load existing results if available
    existing = analyzer.load_existing_results()
    if existing:
        analyzer.results = existing
        print(f"Loaded existing results for: {list(existing.keys())}\n")
    
    if train_default_only:
        # Only train Default NN
        if 'Default_Parking' not in analyzer.results:
            print("Training Default NN only...")
            analyzer.train_default_on_mdp(num_epochs=50)
        else:
            print("Default NN already trained.")
        
        # Generate analysis for what we have
        analyzer.generate_visualizations()
        analyzer.generate_quantitative_report()
        
        # Save results
        import pickle
        with open('analysis_results.pkl', 'wb') as f:
            pickle.dump(analyzer.results, f)
        
        return analyzer, analyzer.results
    
    # Full analysis - skip CNN if requested
    if skip_cnn and 'CNN_CIFAR10' not in analyzer.results:
        print("CNN not in results. Analysis will proceed with MDP networks only.\n")
    
    results = analyzer.run_complete_analysis(
        cnn_epochs=20, 
        mdp_epochs=50, 
        skip_training=False,  # Will check for existing results internally
        load_existing=True,   # Load if available
        skip_cnn=skip_cnn     # Skip CNN training if flag set
    )
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()

