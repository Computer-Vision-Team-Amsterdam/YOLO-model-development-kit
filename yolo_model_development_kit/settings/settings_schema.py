from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None
    ai_instrumentation_key: str = None


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
        "objectherkenning_openbare_ruimte",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: str = ""


class PerformanceEvaluationSpec(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    categories_json_path: str = ""
    dataset_name: str = ""
    model_name: str
    ground_truth_image_shape: List[int]
    predictions_image_shape: List[int]
    prediction_labels_rel_path: str = "labels"
    splits: List[str]
    target_classes: List[int]
    sensitive_classes: List[int]
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    plot_pr_curves: bool = True


class TrainingModelParameters(SettingsSpecModel):
    img_size: int = 1024
    batch: Union[float, int] = -1
    epochs: int = 100
    n_classes: int = 3
    name_classes: List[str] = ["person", "license plate", "container"]
    patience: int = 25
    cos_lr: bool = False
    dropout: float = 0.0
    seed: int = 0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


class TrainingPipelineSpec(SettingsSpecModel):
    model_parameters: TrainingModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    sweep_mode: bool = False


class WandbSpec(SettingsSpecModel):
    api_key: str
    mode: str = "disabled"


class YoloModelDevelopmentKitSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    logging: LoggingSpec = LoggingSpec()
    performance_evaluation: PerformanceEvaluationSpec = None
    training_pipeline: TrainingPipelineSpec = None
    wandb: WandbSpec = None
