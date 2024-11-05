from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from pydantic import BaseModel

from yolo_model_development_kit.settings.settings_schema import (
    YoloModelDevelopmentKitSettingsSpec,
)


class YoloModelDevelopmentKitSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = YoloModelDevelopmentKitSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
