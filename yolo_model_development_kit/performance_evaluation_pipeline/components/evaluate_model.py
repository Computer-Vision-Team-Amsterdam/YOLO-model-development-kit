import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from yolo_model_development_kit import settings  # noqa: E402
from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (  # noqa: E402
    ObjectClass,
)
from yolo_model_development_kit.performance_evaluation_pipeline.source import (  # noqa: E402
    YoloEvaluator,
)

azure_logging_configurer = AzureLoggingConfigurer(settings["logging"])
azure_logging_configurer.setup_oor_logging()
logger = logging.getLogger("performance_evaluation")

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="evaluate_model",
    display_name="Evaluate model predictions.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=True,
)
def evaluate_model(
    ground_truth_base_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Run evaluation of a model based on ground truth annotations and model
    predictions.

    This pipeline runs three evaluation methods:

    * Total Blurred Area evaluation for sensitive classes. This tells us the
      percentage of bounding boxes that are covered by predictions.

    * Per Image evaluation. This tells us the precision and recall based on
      whole images, i.e. if a single image contains at least one annotation of a
      certain class, does it also contain at least one prediction.

    * Custom COCO evaluation. This is a COCO-style evaluation of overall and per
      class precision and recall, for different bounding box sizes and
      confidence thresholds.

    Results are stored as CSV files in the chosen output location.

    Parameters
    ----------

    ground_truth_base_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of ground truth dataset (root folder, is expected to contain
        `images/` and `labels/` subfolders).
    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of predictions (root folder, is expected to contain `labels/`
        subfolder).
    output_dir: Output(type=AssetTypes.URI_FOLDER)
        Location where output will be stored.
    """

    eval_settings = settings["performance_evaluation"]
    categories_json_path = eval_settings["categories_json_path"]
    grouping_json_path = eval_settings["grouping_json_path"]
    dataset_name = eval_settings["dataset_name"]
    model_name = eval_settings["model_name"]
    ground_truth_img_shape = eval_settings["ground_truth_image_shape"]
    predictions_img_shape = eval_settings["predictions_image_shape"]
    prediction_labels_rel_path = eval_settings["prediction_labels_rel_path"]
    splits = eval_settings["splits"]
    target_classes = eval_settings["target_classes"]
    sensitive_classes = eval_settings["sensitive_classes"]
    target_classes_conf = eval_settings["target_classes_conf"]
    sensitive_classes_conf = eval_settings["sensitive_classes_conf"]
    use_groupings = eval_settings.get("use_groupings", False)
    group_types = eval_settings.get("group_types", [])

    logger.info(f"Running performance evaluation for model: {model_name}")
    logger.info(f"Use groupings: {use_groupings}")
    logger.info(f"Group types: {group_types}")

    os.makedirs(output_dir, exist_ok=True)

    # Load categories JSON file once
    ObjectClass.load_categories(categories_json_path)

    if use_groupings:
        logger.info("Applying groupings...")
        ObjectClass.apply_groupings(grouping_json_path, group_types)
        group_types_to_evaluate = group_types
    else:
        # If no groupings, just use the original categories
        logger.info("Loading original categories...")
        ObjectClass.load_categories(categories_json_path)
        group_types_to_evaluate = ["no_grouping"]

    for group_type in group_types_to_evaluate:
        logger.info(f"Starting evaluation for group type: {group_type}")
        if group_type == "no_grouping":
            # Use original categories for non-grouped evaluation
            ObjectClass.load_categories(categories_json_path)
            current_model_name = f"{model_name}_no_grouping"
        else:
            # Load categories for the current group
            logger.info(f"Loading grouped categories for group type: {group_type}")
            grouped_categories = ObjectClass.get_group(group_type)
            ObjectClass.load_categories_from_dict(grouped_categories)
            logger.info(
                f"Loaded categories: {ObjectClass.all_names()}, {ObjectClass.all_ids()}"
            )
            current_model_name = f"{model_name}_{group_type}"

        yolo_eval = YoloEvaluator(
            ground_truth_base_folder=ground_truth_base_dir,
            predictions_base_folder=predictions_base_dir,
            output_folder=output_dir,
            ground_truth_image_shape=ground_truth_img_shape,
            predictions_image_shape=predictions_img_shape,
            dataset_name=dataset_name,
            model_name=current_model_name,
            pred_annotations_rel_path=prediction_labels_rel_path,
            splits=splits,
            target_classes=target_classes,
            sensitive_classes=sensitive_classes,
            target_classes_conf=target_classes_conf,
            sensitive_classes_conf=sensitive_classes_conf,
        )

        logger.info(f"Target classes: {yolo_eval.target_classes}")
        logger.info(f"Sensitive classes: {yolo_eval.sensitive_classes}")
        logger.info(f"Loaded categories IDs: {ObjectClass.all_ids()}")
        logger.info(f"Loaded thresholds: {ObjectClass.all_thresholds()}")

        # Total Blurred Area evaluation
        if len(sensitive_classes) > 0:
            tba_results = yolo_eval.evaluate_tba()
            yolo_eval.save_tba_results_to_csv(results=tba_results)
            # Plot precision/recall curves
            if eval_settings["plot_pr_curves"]:
                yolo_eval.plot_tba_pr_f_curves(show_plot=False)

        # Per Image evaluation
        if len(target_classes) > 0:
            per_image_results = yolo_eval.evaluate_per_image()
            yolo_eval.save_per_image_results_to_csv(results=per_image_results)
            # Plot precision/recall curves
            if eval_settings["plot_pr_curves"]:
                yolo_eval.plot_per_image_pr_f_curves(show_plot=False)

        # Custom COCO evaluation
        if (len(sensitive_classes) > 0) or (len(target_classes) > 0):
            coco_results = yolo_eval.evaluate_coco()
            yolo_eval.save_coco_results_to_csv(results=coco_results)
