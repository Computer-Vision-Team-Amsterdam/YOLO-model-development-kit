import logging
import os
import sys
from typing import Optional, Tuple, Union

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component
from ultralytics import YOLO

sys.path.append("../../..")

from yolo_model_development_kit import settings  # noqa: E402

logger = logging.getLogger("model_conversion_pipeline")

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="model_conversion_pipeline",
    display_name="Convert a YOLO model to TensorRT.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def convert_model(
    model_weights_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    print("Starting model conversion step")
    model_conversion_settings = settings["model_conversion"]
    input_model_name = model_conversion_settings["input_model_name"]
    output_model_name = model_conversion_settings["output_model_name"]
    if not output_model_name:
        output_model_name = f"{os.path.splitext(input_model_name)[0]}.engine"

    input_model_path = os.path.join(model_weights_dir, input_model_name)
    output_model_path = os.path.join(model_weights_dir, output_model_name)

    if not os.path.isfile(input_model_path):
        print(f"Cannot convert model because input {input_model_path} is not found.")
        logger.error(
            f"Cannot convert model because input {input_model_path} is not found."
        )
        return
    if (
        os.path.isfile(output_model_path)
        and not model_conversion_settings["overwrite_if_exists"]
    ):
        print(
            "Converted model already exists. Set overwrite_if_exists=True to overwrite existing model."
        )
        logger.error(
            "Converted model already exists. Set overwrite_if_exists=True to overwrite existing model."
        )
        return

    output_model_type = os.path.splitext(output_model_name)[1]
    if output_model_type == ".pt":
        print("Required model is a Torch model, no conversion needed.")
        logger.info("Required model is a Torch model, no conversion needed.")
        return
    if output_model_type != ".engine":
        print(f"Unknown model type: {output_model_type}.")
        logger.error(f"Unknown model type: {output_model_type}.")
        return

    print(f"Converting {input_model_name} to {output_model_name}..")
    logger.info(f"Converting {input_model_name} to {output_model_name}..")
    _convert_model_to_trt(
        model_path=input_model_path,
        image_size=model_conversion_settings["image_size"],
        batch=model_conversion_settings["batch_size"],
    )

    logger.info(f"Model converted and stored in {output_model_path}")


def _convert_model_to_trt(
    model_path: Union[str, os.PathLike],
    image_size: Optional[Union[Tuple[int, int], int]],
    batch: int = 1,
) -> None:
    export_params = {
        "format": "engine",
        "workspace": None,
        "half": True,
        "batch": batch,
    }

    if image_size is not None:
        export_params["imgsz"] = image_size
    else:
        export_params["dynamic"] = True

    print(f"Converting model using: {export_params}")
    logger.debug(f"Converting model using: {export_params}")
    model = YOLO(model_path, task="detect")
    model.export(**export_params)
