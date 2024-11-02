import jax
import jax.numpy as jnp
import time
from functools import partial
# Define the MLP function
@partial(jax.vmap, in_axes=(None, 0))
def mlp(params, x):
    # Iterate over the layers
    for W, b in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    # Output layer without activation
    W, b = params[-1]
    x = jnp.dot(x, W) + b
    return x

# Initialize parameters
def init_params(layer_sizes, key):
    params = []
    num_layers = len(layer_sizes) - 1
    keys = jax.random.split(key, num_layers * 2)
    for i in range(num_layers):
        W_key = keys[2 * i]
        b_key = keys[2 * i + 1]
        W = jax.random.normal(W_key, (layer_sizes[i], layer_sizes[i + 1]))
        b = jax.random.normal(b_key, (layer_sizes[i + 1],))
        params.append((W, b))
    return params

# Set layer sizes (Adjust as needed)
layer_sizes = [128] * 9  # [128, 128, ..., 128]
key = jax.random.PRNGKey(0)

# Split the key for parameters and input to avoid reusing keys
key, params_key, input_key = jax.random.split(key, 3)
params = init_params(layer_sizes, params_key)

# JIT-compile the MLP function
mlp_jit = jax.jit(mlp)

# Generate random input data
x = jax.random.normal(input_key, (1, 128))

# Warm up the JIT compiler
mlp_jit(params, x).block_until_ready()

# Function to time the inference
def time_inference(mlp_func, params, x, num_runs=1):
    start = time.time()
    for _ in range(num_runs):
        y = mlp_func(params, x)
    # Ensure computation completes
    y.block_until_ready()
    end = time.time()
    total_time = end - start
    avg_time = total_time / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time*1e6:.2f} microseconds")
    return avg_time

# Profile the inference speed
avg_time = time_inference(mlp_jit, params, x)

# Lower the JIT-compiled function to get the MLIR module
mlp_lowered = mlp_jit.lower(params, x)
# Specify the dialect and get the textual representation
mlir_module = str(mlp_lowered.compiler_ir())

# Save the MLIR module to a file
with open('mlp.mlir', 'w') as f:
    f.write(mlir_module)