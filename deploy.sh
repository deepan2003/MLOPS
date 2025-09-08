#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Navigate to the app directory
cd /home/ubuntu/MLOPS

# Activate the virtual environment
source venv/bin/activate

# Install DVC S3 support
pip install "dvc[s3]"

# Pull the latest data and models from S3 remote
# AWS credentials will be passed from GitHub secrets
dvc pull

# Stop the old version of the app (if it's running)
# The pkill command will fail if the process isn't found, so we add || true
pkill gunicorn || true

# Start the new version of the app in the background
# We pipe the output to a log file
nohup gunicorn --bind 0.0.0.0:8000 app:app > app.log 2>&1 &

echo "Deployment finished successfully!"

