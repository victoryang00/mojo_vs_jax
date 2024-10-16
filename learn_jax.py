import jax.numpy as jnp
import xgboost as xgb
import numpy as np
import time
# Example input data
data = np.array([[5.1, 3.5, 1.4, 0.2],  # Example features (same shape as training data)
                 [6.2, 3.4, 5.4, 2.3]])

# Convert input data to JAX array
jax_data = jnp.array(data)

# Load your pre-trained XGBoost model
bst = xgb.Booster()
bst.load_model('xgboost_model.json')

# Perform inference using the XGBoost model
# Note that XGBoost expects NumPy array or DMatrix
dtest = xgb.DMatrix(jax_data)  # Convert JAX array to DMatrix
start = time.time()
for i in range(1000):
    preds = bst.predict(dtest)
end = time.time()
print(end - start)
# Output the predictions
print(preds)