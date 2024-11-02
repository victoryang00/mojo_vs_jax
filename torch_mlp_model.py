import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_sizes=[64, 64], output_size=10):
        super(MLP, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return 0

# Create model instance
model = MLP()
for param in model.parameters():
    param.requires_grad = False
    param.data = torch.randn_like(param)
    
example_input = torch.randn(2, 50)  # Batch size of 2

# Trace the model
scripted_model = torch.jit.trace(model, example_input)

# Save the scripted model
scripted_model.save("mlp_scripted_model.pt")
import numpy as np
import torch

# Define the dimensions of the input (e.g., a flattened 28x28 image for an MLP model)
batch_size = 2
input_size = 50  # Adjust to match the MLP's expected input size

# Generate a random input tensor (use values that fit your use case; here, we're using float32)
input_tensor = torch.randint(
    low=0,
    high=100,  # Upper bound (exclusive)
    size=(batch_size, input_size),
    dtype=torch.int8
)

# Convert the tensor to a NumPy array and save as a binary file
input_tensor.numpy().tofile("input_data.bin")

# Save shape information (optional) for loading later
np.array(input_tensor.shape).astype(np.int64).tofile("input_data_shape.bin")

print("Generated input_data.bin with shape:", input_tensor.shape)