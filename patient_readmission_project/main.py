import os
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    Environment, BuildContext, Data, 
    ManagedOnlineEndpoint, ManagedOnlineDeployment, 
    Model, CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes
# ---------------------------------------------------------
# 1. Workspace Connection
# ---------------------------------------------------------
subscription_id = "5cb7be71-07bc-4dae-a01a-26036607d3ed"
resource_group = "rg-patient-readmission"
workspace_name = "ws-patient-readmission"
experiment_name = "patient-readmission-production"
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
root_dir = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------
# 2. Data Ingestion
# ---------------------------------------------------------
diabetes_data_path = "https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv"
data_name = "uci-diabetes-readmission"
diabetes_data = Data(
    path=diabetes_data_path,
    type=AssetTypes.URI_FILE,
    name=data_name,
    version="1",
    description="UCI Diabetes Readmission Dataset"
)
ml_client.data.create_or_update(diabetes_data)
# ---------------------------------------------------------
# 3. Healthcare Environment (STRICT DOCKER BUILD)
# ---------------------------------------------------------
env_name = "healthcare-secure-env"
healthcare_env = Environment(
    name=env_name,
    version="5",   # ⬅️ bump version
    build=BuildContext(path=os.path.join(root_dir, "environment")),
    description="Healthcare env v5"
)
ml_client.environments.create_or_update(healthcare_env)
ml_client.environments.create_or_update(healthcare_env)
print(f"Environment Build Triggered: {env_name}")
# ---------------------------------------------------------
# 4. Pipeline Definition
# ---------------------------------------------------------
src_dir = os.path.join(root_dir, "src")
prep_component = command(
    name="prep_data",
    inputs={"input_data": Input(type="uri_file")},
    outputs={"prepped_data": Output(type="uri_file")},
    code=src_dir,
    command="python prep.py --input_data ${{inputs.input_data}} --clean_data ${{outputs.prepped_data}}",
    environment=f"{env_name}@latest",
)
train_component = command(
    name="train_model",
    inputs={"training_data": Input(type="uri_file")},
    outputs={"model_output": Output(type="mlflow_model")},
    code=src_dir,
    command="python train.py --training_data ${{inputs.training_data}} --target_column 'Y' --model_output ${{outputs.model_output}}",
    environment=f"{env_name}@latest",
)
register_component = command(
    name="register_model",
    inputs={"model_input": Input(type="mlflow_model")},
    code=src_dir,
    command=f"python register.py --model_input ${{{{inputs.model_input}}}} --subscription_id {subscription_id} --resource_group {resource_group} --workspace_name {workspace_name}",
    environment=f"{env_name}@latest",
)
@dsl.pipeline(compute="ci-patient-dev", description="Patient Readmission E2E Pipeline")
def healthcare_pipeline(raw_data):
    prep_node = prep_component(input_data=raw_data)
    train_node = train_component(training_data=prep_node.outputs.prepped_data)
    register_node = register_component(model_input=train_node.outputs.model_output)
    return {"pipeline_model": train_node.outputs.model_output}
pipeline_job = healthcare_pipeline(Input(type="uri_file", path=f"azureml:{data_name}:1"))
# ---------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------
returned_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name=experiment_name)
print(f"Pipeline submitted! View status: {returned_job.studio_url}")
