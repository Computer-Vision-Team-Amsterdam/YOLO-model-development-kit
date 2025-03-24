# YOLO-model-development-kit
Project to develop a YOLO model using AzureML platform. The code supports both YOLOv8 and YOLO11 models.

## Installation
### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/YOLO-model-development-kit.git
```

### 2. Install UV
We use UV as package manager, which can be installed using any method mentioned on [the UV webpage](https://docs.astral.sh/uv/getting-started/installation/).

The easiest option is to use their installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is also possible to use pip:
```bash
pipx install uv
```

Afterwards, uv can be updated using `uv self update`.

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
uv pip install -r pyproject.toml --extra dev [--extra model_export]
```

To update dependencies (e.g. when pyproject.toml dependencies change):

```bash
uv lock --upgrade
uv sync --extra dev
```
    
### 4. Install pre-commit hooks
The pre-commit hooks help to ensure that all committed code is valid and consistently formatted. We use UV to manage pre-commit as well.

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install pre-commit hooks
uv run pre-commit install

# Optional: update pre-commit hooks
uv run pre-commit autoupdate

# Run pre-commit hooks using
uv run .git/hooks/pre-commit
```


## Usage

Modify the `config.yml` to your needs and run the required pipelines. For example:

```bash
# Create AzureML environment
uv run yolo_model_development_kit/create_aml_environment/create_azure_env.py

# Train a YOLO model
uv run yolo_model_development_kit/training_pipeline/submit_training_pipeline.py
```