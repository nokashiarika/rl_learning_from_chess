"""
Feature encoder using linear transformations.

Learns which features are important through matrix multiplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """
    Linear feature encoder that learns feature importance.
    
    Uses matrix multiplication to transform input features
    into learned representations.
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=218):
        """
        Initialize feature encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of actions)
        """
        super(FeatureEncoder, self).__init__()
        
        # Linear transformations (matrix multiplications)
        # W1: (input_dim, hidden_dim) - learns feature importance
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
        
        Returns:
            Output of shape (batch, output_dim)
        
        Matrix operations:
        - h1 = x @ W1^T + b1  (learns which features matter)
        - h2 = h1 @ W2^T + b2 (learns feature interactions)
        - out = h2 @ W3^T + b3 (final prediction)
        """
        # Matrix multiplication: learns feature importance
        h1 = F.relu(self.W1(x))
        h2 = F.relu(self.W2(h1))
        output = self.W3(h2)
        
        return output


class AttentionFeatureEncoder(nn.Module):
    """
    Feature encoder with attention mechanism.
    
    Learns which features to focus on dynamically.
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=218):
        """
        Initialize attention encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super(AttentionFeatureEncoder, self).__init__()
        
        # Query, Key, Value matrices for attention
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Final layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        """
        Forward pass with attention.
        
        Args:
            x: Input features of shape (batch, input_dim)
        
        Returns:
            Output of shape (batch, output_dim)
        
        Attention mechanism:
        - Q = x @ W_q, K = x @ W_k, V = x @ W_v
        - Attention = softmax(Q @ K^T / sqrt(d)) @ V
        """
        # For single feature vector, self-attention
        # In practice, you might want to reshape features into sequence
        batch_size = x.size(0)
        
        # Query, Key, Value
        Q = self.query(x)  # (batch, hidden_dim)
        K = self.key(x)   # (batch, hidden_dim)
        V = self.value(x) # (batch, hidden_dim)
        
        # Attention: Q @ K^T / sqrt(d)
        # For single vector, this is self-attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, hidden_dim)
        
        # Final layers
        output = F.relu(self.fc1(attended))
        output = self.fc2(output)
        
        return output

