from typing import Any, Dict, List, Optional, Tuple, Union

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


class InferenceModelParameters(SettingsSpecModel):
    batch_size: int = 1
    img_size: Union[Tuple[int, int], int] = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class InferenceSAHIParameters(SettingsSpecModel):
    model_type: str = "ultralytics"
    slice_height: int = 2048
    slice_width: int = 2048
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2


class InferencePipelineSpec(SettingsSpecModel):
    model_params: InferenceModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    sahi_params: InferenceSAHIParameters
    target_classes: List[int] = None
    sensitive_classes: List[int] = []
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    output_image_size: Optional[Tuple[int, int]] = None
    save_detection_images: bool = False
    save_detection_labels: bool = True
    save_all_images: bool = False
    use_sahi: bool = False


class ModelConversionPipelineSpec(SettingsSpecModel):
    datastore_path: str
    model_weights_rel_path: str = ""
    input_model_name: str
    output_model_name: Optional[str] = None
    overwrite_if_exists: bool = False
    image_size: Optional[Union[Tuple[int, int], int]] = None
    batch_size: int = 1


class PerformanceEvaluationSpec(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    categories_json_path: str = ""
    mapping_json_path: str = ""
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
    is_bias_analysis: bool = True


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
    inference_pipeline: InferencePipelineSpec = None
    model_conversion: ModelConversionPipelineSpec = None
    performance_evaluation: PerformanceEvaluationSpec = None
    training_pipeline: TrainingPipelineSpec = None
    wandb: WandbSpec = None
