# patient_readmission_project

Unplanned hospital readmissions cost healthcare systems billions annually and are a key metric of patient care quality. This project implements a production-grade MLOps pipeline to predict the 30-day readmission risk of patients. By identifying high-risk individuals before discharge, clinicians can provide targeted interventions to improve outcomes and reduce hospital penalties.

healthcare-readmission-mlops/
├── .github/workflows/       <-- (Optional) For automated MLOps (CI/CD)
├── src/                     <-- Your "Logic" scripts
│   ├── prep.py
│   ├── train.py
│   └── score.py
├── environment/             <-- Your "Infrastructure"
│   ├── Dockerfile
│   └── conda.yaml
├── notebooks/
│   └── exploration.ipynb    <-- Your early EDA (Exploratory Data Analysis)
├── main.py                  <-- The SDK v2 Orchestration script
├── README.md                <-- The "Story" of your project
└── .gitignore               <-- To hide secrets/temp files

