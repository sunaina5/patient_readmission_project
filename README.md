🏥 Patient Readmission Risk Prediction (Azure MLOps)

This project delivers a production-grade, end-to-end MLOps workflow architected on Azure Machine Learning to address the critical healthcare challenge of 30-day patient readmission. By leveraging the UCI Diabetes clinical dataset, I engineered a predictive system designed to transition seamlessly from local development to a scalable cloud environment.

The Journey: From Failure to SUCCESS

    A key part of this project was the transition from local testing to a robust cloud deployment.

    Initial Challenge (The "Exit Code 3" Error): During the first deployment attempt, 
    the Azure ML inference server failed to boot because it defaulted to searching for a file named main.py.
    
    The Fix: I moved to a manual deployment strategy using a deployment.yml configuration file. 
    This allowed me to explicitly map score.py as the entry script, overriding the default and resolving the FileNotFoundError.

    Environment Syncing: I utilized a conda.yml file to ensure the cloud container (Docker) had the exact versions of 
    scikit-learn, pandas, and numpy used during local training.

Official Dataset Source

    Dataset Name: Diabetes 130-US hospitals for years 1999-2008 Data Set.
    Official UCI Link: [Diabetes 130-Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

    The data used in this project is sourced from the UCI Machine Learning Repository. 
    Contains over 101,000 clinical records of diabetic patients across 130 US hospitals, covering 10 years (1999-2008) of care. 
    The goal is to predict 'readmitted,' a multiclass feature indicating if a patient was readmitted in less than 30 days, more than 30 days, or not at all.


🛠️ Tech Stack & Tools

    1. Machine Learning: Python, Scikit-learn (Gradient Boosting Classifier).
    2. Cloud Platform: Azure Machine Learning (SDK v2).
    3. Deployment: Managed Online Endpoints, Docker.
    4. IDE: VS Code with Azure ML Extension.
