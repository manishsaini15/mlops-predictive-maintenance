import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    """
    SageMaker loads the model from /opt/ml/model/
    """
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """
    Converts incoming request into a pandas DataFrame
    """

    if request_content_type == "application/json":
        data = json.loads(request_body)

        # If input is list of dicts
        if isinstance(data, list):
            return pd.DataFrame(data)

        # If input is dict
        if isinstance(data, dict):
            return pd.DataFrame([data])

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Runs prediction
    """
    preds = model.predict(input_data)
    return preds


def output_fn(prediction, response_content_type):
    """
    Converts prediction to JSON response
    """
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction.tolist()})

    raise ValueError(f"Unsupported response content type: {response_content_type}")