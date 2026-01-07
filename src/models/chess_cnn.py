"""
Convolutional Neural Network for chess state representation.

Uses 8x8x12 board tensor as input and applies convolutional layers
to learn spatial patterns (pawn chains, piece coordination, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    """
    CNN for chess position evaluation.
    
    Input: (batch, 12, 8, 8) board tensor
    Output: (batch, n_actions) Q-values for each action
    """
    
    def __init__(self, n_actions=218, hidden_dim=256):
        """
        Initialize CNN.
        
        Args:
            n_actions: Number of possible actions (max legal moves)
            hidden_dim: Hidden layer dimension
        """
        super(ChessCNN, self).__init__()
        
        # Convolutional layers for spatial pattern learning
        # Input: (batch, 12, 8, 8)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)  # (batch, 32, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (batch, 64, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (batch, 128, 8, 8)
        
        # Flatten: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
        
        Returns:
            Q-values of shape (batch, n_actions)
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 8192)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Q-values
        
        return x


class ChessHybridCNN(nn.Module):
    """
    Hybrid CNN that combines board tensor with feature vector.
    
    Uses both spatial (CNN) and hand-crafted features.
    """
    
    def __init__(self, n_actions=218, n_features=4, hidden_dim=256):
        """
        Initialize hybrid CNN.
        
        Args:
            n_actions: Number of possible actions
            n_features: Number of hand-crafted features
            hidden_dim: Hidden layer dimension
        """
        super(ChessHybridCNN, self).__init__()
        
        # CNN branch for board tensor
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Feature branch for hand-crafted features
        self.feature_fc = nn.Linear(n_features, 64)
        
        # Combine both branches
        # CNN output: 128 * 8 * 8 = 8192
        # Feature output: 64
        combined_dim = 8192 + 64
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, board_tensor, features):
        """
        Forward pass.
        
        Args:
            board_tensor: Input tensor of shape (batch, 12, 8, 8)
            features: Feature vector of shape (batch, n_features)
        
        Returns:
            Q-values of shape (batch, n_actions)
        """
        # CNN branch
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # (batch, 8192)
        
        # Feature branch
        f = F.relu(self.feature_fc(features))  # (batch, 64)
        
        # Combine branches
        combined = torch.cat([x, f], dim=1)  # (batch, 8192 + 64)
        
        # Final layers
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        output = self.fc3(combined)
        
        return output

