FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
# Best Practice: Update system packages to patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Install Essential Healthcare/Data Science Libraries
# explicit versions for reproducibility in production


# Use these specific versions to fix the MLflow tracking_uri conflict
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.25.2 \
    mlflow==2.14.1 \
    azureml-mlflow==1.56.0 \
    azureml-inference-server-http \
    azure-ai-ml \
    azure-identity \
    joblib \
    matplotlib
# Set env vars for standardization
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models
