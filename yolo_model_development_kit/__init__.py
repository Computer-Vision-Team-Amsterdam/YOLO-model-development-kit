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
except FileNotFoundError:
    logger.warning(
        "Config file for YoloModelDevelopmentKit not found. If the project was extended this warning can be ignored."
    )
