# YOLO-model-development-kit
Project to develop a YOLO model using AzureML platform. The code supports both YOLOv8 and YOLO11 models.

## Installation
### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/YOLO-model-development-kit.git
```

### 2. Install UV
We use UV as package manager, which can be installed using any method mentioned on [the UV webpage](https://docs.astral.sh/uv/). In principle, however, it is also possible to use PIP to install the required packages.

The easiest option is to use their installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use UV to create a new virtual environment and install the dependencies.

```bash
cd YOLO-model-development-kit

# Create the environment locally in the folder .venv
# Note: model_export requires a CUDA enabled GPU and python <= 3.11
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate 

# Install dependencies
uv pip install -r pyproject.toml --extra dev 
```
    
### 4. Install pre-commit hooks
The pre-commit hooks help to ensure that all committed code is valid and consistently formatted. We use UV to manage pre-commit as well.

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install pre-commit hooks
pre-commit install

# Optional: update pre-commit hooks
pre-commit autoupdate

# Run pre-commit hooks using
bash .git/hooks/pre-commit
```


## Usage

Modify the `config.yml` to your needs and run the required pipelines.

```bash
# Create AzureML environment
python yolo_model_development_kit/create_aml_environment/create_azure_env.py

# Train a YOLO model
python yolo_model_development_kit/training_pipeline/submit_training_pipeline.py
```