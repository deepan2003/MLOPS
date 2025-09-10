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

# 2. Upgrade pip, setuptools, and wheel first as a best practice
echo "Upgrading pip, setuptools, and wheel..."
venv/bin/pip install --upgrade pip setuptools wheel

# 3. Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
venv/bin/pip install -r requirements.txt

# 4. Pull data using the venv's dvc
echo "Pulling data with DVC..."
venv/bin/dvc pull

# 5. Start the Gunicorn server in the background
echo "Starting Gunicorn server..."
nohup venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 src.api.app:app > /dev/null 2>&1 &

echo "🚀 Deployment successful!"