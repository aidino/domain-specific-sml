#!bin/bash

# Install python 3.14
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.14 python3.14-venv python3.14-dev

# Create environment
# python3.14 -m venv .venv
# source .venv/bin/activate

# Installing toolkit
pip install "numpy>=2.0" "pandas>=3.0" matplotlib torch scikit-learn
pip install "transformers[torch]" datasets

# Install torch with GPU
# pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

# Check env
python3.14 --version
