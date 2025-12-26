import os
import joblib
import logging
import json
import numpy as np

def init():
    global model
    logging.info("Initializing patient-readmission-api...")

    try:
        model_root = os.getenv("AZUREML_MODEL_DIR")

        if model_root:
            logging.info(f"Cloud detected. Searching in: {model_root}")
            for root, _, files in os.walk(model_root):
                for f in files:
                    if f.endswith(".pkl") or f.endswith(".joblib"):
                        model_path = os.path.join(root, f)
                        break
        else:
            model_path = "/home/azureuser/cloudfiles/code/patient_readmission_project/model/model.pkl"

        logging.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logging.info("SUCCESS: Model loaded.")

    except Exception as e:
        logging.error(f"Init failed: {str(e)}")
        raise


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        input_data = np.array(data)

        probs = model.predict_proba(input_data)[:, 1]

        return {
            "results": [
                {
                    "probability": round(float(p), 4),
                    "prediction": "High Risk" if p >= 0.5 else "Low Risk"
                }
                for p in probs
            ]
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return {"error": str(e)}