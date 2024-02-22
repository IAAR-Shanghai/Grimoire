#!/bin/bash

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run embed.py
echo "Running embed.py..."
python data/embed.py

# Run compute_similarity.py
echo "Running compute_similarity.py..."
python data/compute_similarity.py

echo "Environment setup and script execution complete."

