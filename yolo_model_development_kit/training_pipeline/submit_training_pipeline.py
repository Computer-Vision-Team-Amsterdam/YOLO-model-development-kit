import os

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from yolo_model_development_kit import aml_interface, settings
from yolo_model_development_kit.training_pipeline.components import (
    sweep_model,
    train_model,
)


@pipeline()
def training_pipeline():

    datastore_path = aml_interface.get_datastore_full_path(
        settings["training_pipeline"]["inputs"]["datastore_path"]
    )
    training_rel_path = settings["training_pipeline"]["inputs"][
        "training_data_rel_path"
    ]
    model_weights_rel_path = settings["training_pipeline"]["inputs"][
        "model_weights_rel_path"
    ]
    project_datastore_path = aml_interface.get_datastore_full_path(
        settings["training_pipeline"]["outputs"]["project_datastore_path"]
    )
    project_rel_path = settings["training_pipeline"]["outputs"]["project_rel_path"]

    training_data_path = os.path.join(datastore_path, training_rel_path)
    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=training_data_path,
    )

    model_weights_path = os.path.join(datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )
    if settings["training_pipeline"]["sweep_mode"]:
        train_model_step = sweep_model(
            mounted_dataset=training_data, model_weights=model_weights
        )
    else:
        train_model_step = train_model(
            mounted_dataset=training_data, model_weights=model_weights
        )
    train_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=model_weights_path
    )

    train_model_step.environment_variables = {
        "WANDB_API_KEY": settings["wandb"]["api_key"],
        "WANDB_MODE": settings["wandb"]["mode"],
    }

    project_path = os.path.join(project_datastore_path, project_rel_path)
    train_model_step.outputs.project_path = Output(
        type="uri_folder", mode="rw_mount", path=project_path
    )

    return {}


def main() -> None:
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface.submit_pipeline_experiment(
        training_pipeline, "training_pipeline", default_compute
    )


if __name__ == "__main__":
    main()
