import os

from yolo_model_development_kit.settings import YoloModelDevelopmentKitSettings

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)

YoloModelDevelopmentKitSettings.set_from_yaml(config_path)
settings = YoloModelDevelopmentKitSettings.get_settings()
