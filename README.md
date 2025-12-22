🏥 Patient Readmission Risk Prediction (Azure MLOps)

📌 Overview

Unplanned hospital readmissions cost healthcare systems billions annually and are a key indicator of patient care quality. This project implements a production-grade, end-to-end MLOps pipeline on Azure Machine Learning (SDK v2) to predict 30-day patient readmission risk.

By identifying high-risk patients before discharge, healthcare providers can apply targeted interventions to improve outcomes and reduce penalties associated with avoidable readmissions.

NOTE: The solution is implemented using Azure ML SDK v2 in code (VS Code), while Azure ML Studio is used for experiment tracking, pipeline visualization, and endpoint management.


🎯 Project Goals
	•	Build a reproducible, scalable ML pipeline using Azure ML
	•	Handle healthcare-specific preprocessing and PII considerations
	•	Train and evaluate a Gradient Boosting model
	•	Track experiments and metrics using MLflow
	•	Deploy the model to a secure managed online endpoint


🧱 Architecture Overview

Raw Data

   ↓

Data Preparation (PII handling, encoding)

   ↓
   
Model Training (Gradient Boosting + MLflow)

   ↓
   
Model Registration (threshold-based)

   ↓
   
Managed Online Endpoint (real-time inference)
