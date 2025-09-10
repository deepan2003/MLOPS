This project demonstrates a complete MLOps workflow for training, versioning, and deploying a machine learning model that predicts heart attack risk. It features an automated CI/CD pipeline that deploys a Flask-based web application to a live AWS EC2 server.

🚀 Live Demo
The application is live and can be tested here:
http://56.228.32.2:8000

Features
Reproducible ML Pipeline: The entire data preprocessing and model training workflow is version-controlled and automated using DVC.

Automated CI/CD: Every git push to the main branch automatically triggers a GitHub Actions workflow to build, test, and deploy the application.

Cloud Deployment: The application is deployed to a live AWS EC2 instance.

Interactive Web API: A Flask application serves the model through both a user-friendly web interface and a REST API endpoint, running on a production-ready Gunicorn server.

Persistent Service: The application runs as a systemd service on the server, ensuring it automatically restarts on reboots or crashes.

Tech Stack
Backend: Python, Flask

ML/Data Science: Scikit-learn, Pandas, NumPy, Joblib

MLOps Tools: DVC, GitHub Actions

Infrastructure: AWS EC2, Gunicorn

Testing: Pytest

MLOps Architecture
This project follows a modern CI/CD approach for machine learning.

DVC Pipeline (dvc.yaml)
DVC defines the sequence of steps to go from raw data to a trained model.

Raw Data -> preprocess.py -> Processed Data -> train_model.py -> Trained Model (.pkl) & Metrics (.json)
CI/CD Deployment Pipeline (GitHub Actions)
This pipeline automates the deployment process.

1. Git Push to `main` branch
   |
   V
2. GitHub Actions Workflow Triggered
   |--> Job 1: Build & Test (Reproduce DVC pipeline, run tests)
   |
   V
3. Job 2: Deploy to AWS EC2
   |--> Connect via SSH
   |--> Sync latest project files
   |--> Execute deploy.sh script on server
       |--> Create venv
       |--> Install dependencies
       |--> DVC pull (download model)
       |--> Start Gunicorn service
How to Run Locally
To set up and run this project on your local machine, follow these steps.

Clone the repository:

Bash

git clone https://github.com/deepan2003/MLOPS.git
cd MLOPS
Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the dependencies:

Bash

pip install -r requirements.txt
Pull the data and model from DVC remote storage:

Bash

dvc pull
Run the Flask application locally:

Bash

python src/api/app.py
The application will be available at http://127.0.0.1:5000.


├── dvc.yaml                # DVC pipeline definition file
├── params.yaml             # Parameters for the DVC pipeline
└── requirements.txt        # Python dependencies
