import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the dimensions
input_size = 512  # Size of the encoded visual features
hidden_size = 256  # Size of the hidden layer in MLP
output_size = 4  # Number of high-level actions

# Create the high-level policy head
high_level_policy = MLP(input_size, hidden_size, output_size)

# Softmax layer for action probabilities
softmax = nn.Softmax(dim=1)  # Corrected dim argument

# Example usage:
encoded_visual_features = torch.randn(2, input_size)  # Example input
action_logits = high_level_policy(encoded_visual_features)  # Forward pass
action_probs = softmax(action_logits)  # Apply softmax to get action probabilities

# Simple greedy sampling of high-level actions
_, high_level_actions = torch.max(action_probs, dim=1)  # Corrected dim argument

print("Action Logits:")
print(action_logits)
print("Action Probabilities:")
print(action_probs)
print("High-Level Actions:")
print(high_level_actions)