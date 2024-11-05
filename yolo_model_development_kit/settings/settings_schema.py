from typing import Any, Dict, List, Optional

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


class YoloModelDevelopmentKitSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    logging: LoggingSpec = LoggingSpec()
    performance_evaluation: PerformanceEvaluationSpec = None
