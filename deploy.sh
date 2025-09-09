#!/bin/bash

# Stop any existing application process
pkill gunicorn

# Set the project directory
PROJECT_DIR="/home/ubuntu/MLOPS"
cd $PROJECT_DIR

# 1. Create a new Python virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt

# 4. Pull the latest data from DVC
dvc pull

# 5. Start the Gunicorn server in the background
nohup gunicorn --workers 3 --bind 0.0.0.0:8000 app:app &

echo "🚀 Deployment successful!"