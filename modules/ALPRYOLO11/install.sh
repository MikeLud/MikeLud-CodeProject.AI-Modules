#!/bin/bash

# Installation script for ALPR (YOLOv8) module
#
# This script is called from the ALPR directory using:
#
#    bash ../../CodeProject.AI-Server/src/setup.sh
#
# The setup.sh script will find this install.sh file and execute it.

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../CodeProject.AI-Server/src/setup.sh"
    echo
    exit 1 
fi

# Create directories if they don't exist
mkdir -p "models"
mkdir -p "models/onnx"
mkdir -p "test"

# Copy alpr_system_v205.py to the directory with appropriate naming
cp "${moduleDirPath}/alpr-system-v205.py" "${moduleDirPath}/alpr_system_v205.py"

# For Jetson, we need to install Torch before the other packages.
if [ "$moduleInstallErrors" = "" ] && [ "$edgeDevice" = "Jetson" ]; then 

    # Dependencies (Libraries)
    installAptPackages "python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev zlib1g-dev"
    
    # Dependencies (Python packages)
    installPythonPackagesByName "future wheel mock testresources setuptools==58.3.0 Cython"
    
    # Jetson-specific torch setup would go here
    # ...
fi

# Install dependencies needed for OpenCV on Linux
if [ "$moduleInstallErrors" = "" ] && [ "$inDocker" != true ] && [ "$os" = "linux" ] ; then
    installAptPackages "libgl1-mesa-glx libglib2.0-0"
fi

# Download the ONNX models and store in /models/onnx
if [ "$moduleInstallErrors" = "" ]; then
    getFromServer "models/" "alpr-models-yolov8-onnx.zip" "models/onnx" "Downloading ALPR YOLOv8 ONNX models..."
fi

# Download a test image
if [ "$moduleInstallErrors" = "" ]; then
    getFromServer "test/" "license_plate_test.jpg" "test" "Downloading test image..."
fi