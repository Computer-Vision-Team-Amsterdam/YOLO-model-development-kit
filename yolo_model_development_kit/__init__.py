import os

from aml_interface.aml_interface import AMLInterface

from yolo_model_development_kit.settings import YoloModelDevelopmentKitSettings

aml_interface = AMLInterface()

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
YoloModelDevelopmentKitSettings.set_from_yaml(config_path)
settings = YoloModelDevelopmentKitSettings.get_settings()

__all__ = [aml_interface, settings]
