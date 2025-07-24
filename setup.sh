#!/bin/bash

# Mesh Distance Comparison Tool - Setup Script
# This script sets up the virtual environment and installs dependencies

set -e  # Exit on any error

VENV_NAME="venv"
PYTHON_VERSION="python3.10"

echo "=== Mesh Distance Comparison Tool Setup ==="

# Check if Python 3 is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Using Python: $(which $PYTHON_VERSION)"
echo "Python version: $($PYTHON_VERSION --version)"

# Check if virtual environment exists
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_VERSION -m venv "$VENV_NAME"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python3.10 -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements_mesh_comparison.txt..."
python3.10 -m pip install -r requirements_mesh_comparison.txt

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the mesh comparison tool:"
echo "  1. Activate the virtual environment: source $VENV_NAME/bin/activate"
echo "  2. Run the script: python mesh_distance_comparison.py"
echo ""
echo "Or use the run script: ./run.sh"
echo "" 
