#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Stop any existing application process
pkill gunicorn || true

# Set the project directory
PROJECT_DIR="/home/ubuntu/MLOPS"
cd $PROJECT_DIR

# 1. Create a new Python virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt

# 4. Pull the latest data from DVC
dvc pull

# 5. Start the Gunicorn server in the background
nohup gunicorn --workers 3 --bind 0.0.0.0:8000 app:app &

echo "🚀 Deployment successful!"