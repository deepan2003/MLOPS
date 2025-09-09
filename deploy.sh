#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Stop any existing application process
pkill gunicorn || true

# Set the project directory
PROJECT_DIR="/home/ubuntu/MLOPS"
cd $PROJECT_DIR

# 1. Create a new Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# 2. Install packages using the venv's pip (This is the fix)
echo "Installing packages from requirements.txt..."
venv/bin/pip install -r requirements.txt

# 3. Pull data using the venv's dvc
echo "Pulling data with DVC..."
venv/bin/dvc pull

# 4. Start the server using the venv's gunicorn
echo "Starting Gunicorn server..."
nohup venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 app:app &

echo "🚀 Deployment successful!"