import jax
import jax.numpy as jnp
import time

# Define the MLP function without loops for static unrolling
def mlp(params, x):
    W1, b1, W2, b2, W3, b3, W4, b4 = params

    x = jnp.dot(x, W1) + b1
    x = jax.nn.relu(x)

    x = jnp.dot(x, W2) + b2
    x = jax.nn.relu(x)

    x = jnp.dot(x, W3) + b3
    x = jax.nn.relu(x)

    x = jnp.dot(x, W4) + b4
    return x

# Initialize parameters with static shapes
def init_params(key):
    keys = jax.random.split(key, 8)
    W1 = jax.random.normal(keys[0], (64, 128))
    b1 = jax.random.normal(keys[1], (128,))
    W2 = jax.random.normal(keys[2], (128, 128))
    b2 = jax.random.normal(keys[3], (128,))
    W3 = jax.random.normal(keys[4], (128, 128))
    b3 = jax.random.normal(keys[5], (128,))
    W4 = jax.random.normal(keys[6], (128, 64))
    b4 = jax.random.normal(keys[7], (64,))
    return (W1, b1, W2, b2, W3, b3, W4, b4)

# Initialize parameters
key = jax.random.PRNGKey(0)
params = init_params(key)

# JIT-compile the MLP function with static arguments
mlp_jit = jax.jit(mlp, static_argnums=())

# Generate random input data with static shape
x = jax.random.normal(key, (1, 64))

# Warm up the JIT compiler
mlp_jit(params, x).block_until_ready()

# Function to time the inference
def time_inference(mlp_func, params, x, num_runs=1):
    start = time.time()
    for i in range(num_runs):
        y = mlp_func(params, x[i])
    y.block_until_ready()  # Ensure computation completes
    end = time.time()
    total_time = end - start
    avg_time = total_time / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time*1e6:.2f} microseconds")
    return avg_time

x = [jax.random.normal(key, (1, 64)) for i in range(1)]
# Profile the inference speed
avg_time = time_inference(mlp_jit, params, x)
