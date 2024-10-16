import xgboost as xgb
import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training and testing data to DMatrix, which is XGBoost's data format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# Assume you have trained an XGBoost model named `bst`
# Train your model
# Set the parameters for the XGBoost model
params = {
    'objective': 'multi:softprob',  # Multiclass classification problem
    'num_class': 500,  # Number of classes in the target variable (3 for iris dataset)
    'max_depth': 5,  # Maximum depth of the trees
    'learning_rate': 0.1,  # Learning rate
    'n_estimators': 100,  # Number of boosting rounds
    'eval_metric': 'mlogloss'  # Evaluation metric: multi-class log loss
}

# Train the model using xgb.train()
bst = xgb.train(params, dtrain, num_boost_round=100)
# Save the model to a file
bst.save_model('xgboost_model.json')
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')

# Perform predictions
preds = loaded_model.predict(dtest)
print(preds)