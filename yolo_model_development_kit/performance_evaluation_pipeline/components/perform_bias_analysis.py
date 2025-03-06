import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from yolo_model_development_kit import settings  # noqa: E402
from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (  # noqa: E402
    CategoryManager,
)
from yolo_model_development_kit.performance_evaluation_pipeline.source import (  # noqa: E402
    YoloEvaluator,
    log_label_counts,
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
    ground_truth_labels_rel_path = eval_settings["ground_truth_labels_rel_path"]
    logger.info(f"Running bias analysis for model: {eval_settings['model_name']}")

    os.makedirs(output_dir, exist_ok=True)

    category_manager = CategoryManager(
        categories_json_path=eval_settings["categories_json_path"],
        mappings_json_path=eval_settings["mapping_json_path"],
    )

    logger.info(f"Loaded categories IDs: {category_manager.all_ids()}")
    logger.info(f"Loaded thresholds: {category_manager.all_thresholds()}")
    logger.info(f"Loaded groupings: {category_manager.all_groupings()}")

    original_gt_labels_path = os.path.join(
        ground_truth_base_dir, ground_truth_labels_rel_path
    )
    logger.info(f"Original ground truth labels path: {original_gt_labels_path}")
    groupings = category_manager.all_groupings()

    for grouping in groupings:
        grouping = category_manager.get_grouping(grouping)
        group_name = grouping["group_name"]
        group_id = grouping["group_id"]

        target_classes = grouping["maps_to"].get("class", [])
        if not target_classes or len(target_classes) != 1:
            raise ValueError(
                f"Invalid 'maps_to' for grouping '{group_name}'. Expected exactly one target class, got: {target_classes}."
            )
        maps_to_class = int(target_classes[0])
        logger.info(f"Processing grouping: {grouping}")
        logger.info(f"Grouping_name: {group_name}. Group ID: {group_id}")
        logger.info(
            f"Mapping classes in the group {group_name} to class: {maps_to_class}"
        )

        new_labels_path = os.path.join(ground_truth_base_dir, group_name)
        logger.info(f"Creating new labels folder: {new_labels_path}")
        os.makedirs(new_labels_path, exist_ok=True)

        category_mapping = category_manager.get_category_mapping(group_name)
        logger.info(f"Category mapping: {category_mapping}")

        process_labels(
            original_gt_labels=original_gt_labels_path,
            ground_truth_rel_path=ground_truth_labels_rel_path,
            new_gt_labels_path=new_labels_path,
            category_mapping=category_mapping,
        )

        valid_category_ids = set(category_mapping.values())
        logger.info(
            f'Valid category IDs for grouping "{group_name}": {valid_category_ids}'
        )
        log_label_counts(new_labels_path, logger, valid_category_ids)

        current_ground_truth_base_dir = new_labels_path

        yolo_eval = YoloEvaluator(
            ground_truth_base_folder=current_ground_truth_base_dir,
            predictions_base_folder=predictions_base_dir,
            category_manager=category_manager,
            output_folder=output_dir,
            predictions_image_shape=eval_settings["predictions_image_shape"],
            dataset_name=eval_settings["dataset_name"],
            model_name=eval_settings["model_name"],
            pred_annotations_rel_path=eval_settings["prediction_labels_rel_path"],
            gt_annotations_rel_path=ground_truth_labels_rel_path,
            splits=eval_settings["splits"],
            target_classes=[maps_to_class],
            sensitive_classes=eval_settings["sensitive_classes"],
            target_classes_conf=eval_settings["target_classes_conf"],
            sensitive_classes_conf=eval_settings["sensitive_classes_conf"],
        )

        # Total Blurred Area evaluation
        tba_results = yolo_eval.evaluate_tba_bias_analysis(grouping=grouping)
        yolo_eval.save_tba_results_to_csv(
            results=tba_results, use_groupings=True, group_id=group_id
        )
