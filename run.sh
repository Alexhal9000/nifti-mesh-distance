#!/bin/bash

# Mesh Distance Comparison Tool - Run Script
# This script activates the virtual environment and runs the comparison tool

set -e  # Exit on any error

VENV_NAME="venv"

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Error: Virtual environment '$VENV_NAME' not found."
    echo "Please run './setup.sh' first to set up the environment."
    exit 1
fi

# Check if the main script exists
if [ ! -f "mesh_distance_comparison.py" ]; then
    echo "Error: mesh_distance_comparison.py not found."
    exit 1
fi

echo "=== Running Mesh Distance Comparison Tool ==="
echo "Activating virtual environment..."

# Activate virtual environment and run the script
source "$VENV_NAME/bin/activate"
python3.10 mesh_distance_comparison.py

echo "=== Tool execution completed ===" 
