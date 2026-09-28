import logging
import os

from aml_interface.aml_interface import AMLInterface

from yolo_model_development_kit.settings import YoloModelDevelopmentKitSettings

logger = logging.getLogger("inference_pipeline")

aml_interface = AMLInterface()

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
try:
    YoloModelDevelopmentKitSettings.set_from_yaml(config_path)
    settings = YoloModelDevelopmentKitSettings.get_settings()

    aml_env_string = aml_interface.get_aml_environment_string(
        env_name=settings["aml_experiment_details"]["env_name"],
        env_version=settings["aml_experiment_details"]["env_version"],
    )
except FileNotFoundError:
    logger.warning(
        "Config file for YoloModelDevelopmentKit not found. If the project was extended this warning can be ignored."
    )
