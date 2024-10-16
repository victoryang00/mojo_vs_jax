from python import Python
from max import engine
from max.tensor import TensorSpec, TensorShape
from pathlib import Path

def main():
    # Load your model:
    model_path =  Path("xgboost_model.onnx")
    np = Python.import_module("numpy")
    session = engine.InferenceSession()
    model = session.load(model_path)

    # Get the inputs, then run an inference:
    # Define X_test with some sample data
    X_test = Python.list()
    X_test1 = Python.list()
    X_test2 = Python.list()
    X_test1.append(1)
    X_test1.append(2)
    X_test1.append(3)
    X_test1.append(4)
    X_test1.append(5)
    X_test1.append(6)
    X_test1.append(7)
    X_test2.append(8)
    X_test.append(X_test1)
    X_test.append(X_test2)
    test_data = np.array(X_test, dtype=np.float32)
    for i in range(1000):
        outputs = model.execute(test_data)
    # Process the output here.
    