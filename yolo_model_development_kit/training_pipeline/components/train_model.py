import json
import os
import sys
from typing import Dict, Optional

import wandb
import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from ultralytics import YOLO
from ultralytics import settings as ultralytics_settings
from wandb.integration.ultralytics import add_wandb_callback

sys.path.append("../../..")

from yolo_model_development_kit import settings  # noqa: E402

aml_experiment_settings = settings["aml_experiment_details"]


def load_training_parameters(json_file: Optional[str]) -> Dict:
    """
    Load parameters from a JSON file and return them as a dictionary. The method
    will return an empty dictionary if the file is not provided, is empty or it
    doesn't exist.

    Parameters
    ----------
    json_file: Optional[str]
        Path to JSON file.

    Returns
    -------
    Dictionary with loaded parameters.
    """
    if not json_file:  # If no config file is provided
        return {}

    if not os.path.exists(json_file):  # Check if file exists
        return {}

    if os.stat(json_file).st_size == 0:  # Check if file is empty
        return {}

    try:
        with open(json_file, "r") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {json_file}: {e}")

    parameters = {k: v["value"] for k, v in config.get("parameters", {}).items()}
    return parameters


@command_component(
    name="train_model",
    display_name="Train a YOLO model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def train_model(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    project_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to train a YOLO model.

    Parameters
    ----------
    mounted_dataset:
        Dataset to use for training, it should contain the following folder structure:
            - /images/train/
            - /images/val/
            - /images/test/
            - /labels/train/
            - /labels/val/
            - /labels/test/
    model_weights:
        Path to the pretrained model weights.
    yolo_yaml_path:
        Location where to store the yaml file for yolo training.
    project_path:
        Location where to store the outputs of the model.
    """

    ultralytics_settings.update({"runs_dir": project_path})
    wandb.init(
        job_type="training",
        entity=settings["wandb"]["entity"],
        project=settings["wandb"]["project_name"],
    )

    # Load parameters from the JSON configuration file, if provided
    config_file = settings["training_pipeline"]["inputs"].get("config_file", None)
    train_params_from_json = load_training_parameters(config_file)

    n_classes = settings["training_pipeline"]["model_parameters"]["n_classes"]
    name_classes = settings["training_pipeline"]["model_parameters"]["name_classes"]
    data = dict(
        path=f"{mounted_dataset}",
        train="images/train/",
        val="images/val/",
        test="images/test/",
        nc=n_classes,
        names=name_classes,
    )
    yaml_path = os.path.join(yolo_yaml_path, f"oor_dataset_cfg_nc_{n_classes}.yaml")
    with open(f"{yaml_path}", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    model_name = settings["training_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    model_parameters = settings["training_pipeline"]["model_parameters"]

    project_name = settings["training_pipeline"]["outputs"]["project_name"]
    if project_name == "":
        project_name = None

    # The batch_size can be a float between 0 and 1, or a positive int.
    # This is ambiguous in the config.yml parsing, so we need to fix it here.
    batch_size = model_parameters.get("batch", -1)
    if (batch_size >= 1) and (isinstance(batch_size, float)):
        batch_size = int(batch_size)

    model = YOLO(model=pretrained_model_path, task="detect")

    # Add W&B Callback for Ultralytics
    add_wandb_callback(
        model,
        enable_model_checkpointing=False,
        enable_validation_logging=False,
        enable_prediction_logging=False,
        enable_train_validation_logging=False,
    )

    # Prepare parameters for training
    train_params = {
        "data": yaml_path,
        "epochs": model_parameters.get("epochs", 100),
        "imgsz": model_parameters.get("img_size", 1024),
        "project": project_path,
        "save_dir": project_path,
        "name": project_name,
        "batch": batch_size,
        "patience": model_parameters.get("patience", 100),
        "cos_lr": model_parameters.get("cos_lr", False),
        "dropout": model_parameters.get("dropout", 0.0),
        "seed": model_parameters.get("seed", 0),
        "box": model_parameters.get("box", 7.5),
        "cls": model_parameters.get("cls", 0.5),
        "dfl": model_parameters.get("dfl", 1.5),
    }
    train_params.update(train_params_from_json)  # Update with dynamically loaded params

    # Train the model
    model.train(**train_params)
