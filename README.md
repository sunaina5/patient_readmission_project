🏥 Patient Readmission Risk Prediction (Azure MLOps)


    2. The Journey: From Failure to Success
    A key part of this project was the transition from local testing to a robust cloud deployment.

    Initial Challenge (The "Exit Code 3" Error): During the first deployment attempt, the Azure ML inference server failed to boot because it defaulted to searching for a file named main.py.
    The Fix: I moved to a manual deployment strategy using a deployment.yml configuration file. This allowed me to explicitly map score.py as the entry script, overriding the default and resolving the FileNotFoundError.

    Environment Syncing: I utilized a conda.yml file to ensure the cloud container (Docker) had the exact versions of scikit-learn, pandas, and numpy used during local training.
