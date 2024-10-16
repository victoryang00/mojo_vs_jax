import xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx import update_registered_converter
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
xgb_clf = xgboost.XGBClassifier(n_estimators=100, max_depth=5, objective='rank:ndcg', learning_rate=0.1)
xgb_clf.fit(X_train, y_train)

# Register a shape calculator and a converter for XGBoostClassifier
update_registered_converter(
    xgboost.XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes,
    convert_xgboost
)

# Define the input type for the model
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

onnx_model = convert_sklearn(xgb_clf, initial_types=initial_type,target_opset={'ai.onnx.ml': 1})

# Save the ONNX model
with open("xgboost_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model has been converted to ONNX format and saved as xgboost_model.onnx")
