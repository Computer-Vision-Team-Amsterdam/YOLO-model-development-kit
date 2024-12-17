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
    process_labels,
)

azure_logging_configurer = AzureLoggingConfigurer(settings["logging"])
azure_logging_configurer.setup_oor_logging()
logger = logging.getLogger("performance_evaluation")

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="perform_bias_analysis",
    display_name="Perform Bias Analysis on a model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=True,
)
def perform_bias_analysis(
    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    ground_truth_base_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Run bias analysis of a model based on ground truth annotations, model
    predictions and a set of groupings provided by a JSON file.

    This pipeline runs one evaluation method:

    * Total Blurred Area evaluation for sensitive classes. This tells us the
      percentage of bounding boxes that are covered by predictions.

    Results are stored as CSV files in the chosen output location.

    Parameters
    ----------

    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of predictions (root folder, is expected to contain `labels/`
        subfolder).
    ground_truth_base_dir: Output(type=AssetTypes.URI_FOLDER)
        Location of ground truth dataset (root folder, is expected to contain
        `images/` and `labels/` subfolders).
    output_dir: Output(type=AssetTypes.URI_FOLDER)
        Location where output will be stored.
    """

    eval_settings = settings["performance_evaluation"]

    logger.info(f"Running bias analysis for model: {eval_settings["model_name"]}")

    os.makedirs(output_dir, exist_ok=True)

    ObjectClass.load_categories(eval_settings["categories_json_path"])
    ObjectClass.load_mapping(eval_settings["mapping_json_path"])

    logger.info(f"Loaded categories IDs: {ObjectClass.all_ids()}")
    logger.info(f"Loaded thresholds: {ObjectClass.all_thresholds()}")
    logger.info(f"Loaded groupings: {ObjectClass.all_groupings()}")

    groupings = ObjectClass.all_groupings()

    original_gt_labels_path = os.path.join(ground_truth_base_dir, "labels")
    logger.info(f"Original ground truth labels path: {original_gt_labels_path}")

    for grouping in groupings:

        grouping = ObjectClass.get_grouping(grouping)
        group_name = grouping["group_name"]
        maps_to_class = int(
            grouping["maps_to"]["class"][0]
        )  # Assuming there's always one target class in the "maps_to"
        logger.info(f"Processing grouping: {grouping}")
        logger.info(f"Grouping_name: {group_name}")

        new_labels_path = os.path.join(ground_truth_base_dir, group_name)
        logger.info(f"Creating new labels folder: {new_labels_path}")
        os.makedirs(new_labels_path, exist_ok=True)

        category_mapping = ObjectClass.get_category_mapping(group_name)
        logger.info(f"Category mapping: {category_mapping}")

        process_labels(
            original_gt_labels=original_gt_labels_path,
            new_gt_labels_path=new_labels_path,
            category_mapping=category_mapping,
        )

        current_ground_truth_base_dir = new_labels_path

        yolo_eval = YoloEvaluator(
            ground_truth_base_folder=current_ground_truth_base_dir,
            predictions_base_folder=predictions_base_dir,
            output_folder=output_dir,
            ground_truth_image_shape=eval_settings["ground_truth_image_shape"],
            predictions_image_shape=eval_settings["predictions_image_shape"],
            dataset_name=eval_settings["dataset_name"],
            model_name=eval_settings["model_name"],
            pred_annotations_rel_path=eval_settings["prediction_labels_rel_path"],
            splits=eval_settings["splits"],
            target_classes=[
                maps_to_class
            ],  # Only take the target class the grouping maps to
            sensitive_classes=eval_settings["sensitive_classes"],
            target_classes_conf=eval_settings["target_classes_conf"],
            sensitive_classes_conf=eval_settings["sensitive_classes_conf"],
        )

        # Total Blurred Area evaluation
        tba_results = yolo_eval.evaluate_tba_bias_analysis(grouping=grouping)
        yolo_eval.save_tba_results_to_csv(results=tba_results, use_groupings=True)
