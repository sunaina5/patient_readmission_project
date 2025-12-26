import argparse
import mlflow
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True, help="Path to trained model artifact")
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    return parser.parse_args()
def main(args):
    print(f"Registering model from {args.model_input}")
    
    try:
        # Use Managed Identity of the Compute Instance/Cluster
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential, 
            args.subscription_id, 
            args.resource_group, 
            args.workspace_name
        )
        
        # Check if model folder contains the model file
        model_path = args.model_input
        # If mlflow saved as a folder, we register the folder (ModelType.MLFLOW_MODEL)
        # or the file (ModelType.CUSTOM)
        
        print("Registering model asset...")
        model = Model(
            path=model_path,
            type=AssetTypes.MLFLOW_MODEL,
            name="readmission-model",
            description="Patient Readmission Prediction Model (Gradient Boosting)",
            tags={"framework": "scikit-learn", "data": "diabetes"}
        )
        
        registered_model = ml_client.models.create_or_update(model)
        print(f"Model registered: {registered_model.name} version {registered_model.version}")
        
    except Exception as e:
        print(f"Failed to register model: {e}")
        # Investigate permissions if this fails
        raise e
if __name__ == "__main__":
    main(parse_args())
