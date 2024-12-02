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
    mapping_json_path = eval_settings["mapping_json_path"]
    dataset_name = eval_settings["dataset_name"]
    model_name = eval_settings["model_name"]
    ground_truth_img_shape = eval_settings["ground_truth_image_shape"]
    predictions_img_shape = eval_settings["predictions_image_shape"]
    prediction_labels_rel_path = eval_settings["prediction_labels_rel_path"]
    splits = eval_settings["splits"]
    sensitive_classes = eval_settings["sensitive_classes"]
    target_classes_conf = eval_settings["target_classes_conf"]
    sensitive_classes_conf = eval_settings["sensitive_classes_conf"]

    logger.info(f"Running bias analysis for model: {model_name}")

    os.makedirs(output_dir, exist_ok=True)

    # Load categories JSON file once
    ObjectClass.load_categories(categories_json_path)

    # Load mapping JSON file once
    ObjectClass.load_mapping(mapping_json_path)

    logger.info(f"Loaded categories IDs: {ObjectClass.all_ids()}")
    logger.info(f"Loaded thresholds: {ObjectClass.all_thresholds()}")
    logger.info(f"Loaded groupings: {ObjectClass.all_groupings()}")

    # For loop for each grouping in mapping JSON
    # TODO: Implement this in the future

    groupings = ObjectClass.all_groupings()

    # Map category IDs to ground truth labels and make a new labels folder
    # 1. Map original class IDs to the corresponding new category IDs based on the grouping specified
    def process_label(label_path, new_labels_path, category_mapping):
        # Read the label file
        with open(label_path, "r") as lf:
            lines = lf.readlines()

        # Process each line and map the class ID
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            original_class_id = int(parts[0])

            # Map to new category ID
            if original_class_id in category_mapping:
                new_class_id = category_mapping[original_class_id]
                parts[0] = str(new_class_id)
            else:
                # If no mapping is found, skip or keep the original ID
                new_class_id = original_class_id

            # Rebuild the line with updated class ID
            new_lines.append(" ".join(parts))

        # Write back the updated labels
        new_label_path = os.path.join(new_labels_path, os.path.basename(label_path))
        with open(new_label_path, "w") as lf:
            lf.write("\n".join(new_lines))

    def process_labels(original_gt_labels, new_gt_labels_path, category_mapping):
        """
        Processes labels and copies them from a dataset to an output dataset.
        """
        # List all labels (.txt files) in the dataset labels folder
        labels_list = os.listdir(original_gt_labels)
        print(len(labels_list))
        new_gt_labels_path = os.path.join(new_gt_labels_path, "labels")
        os.makedirs(new_gt_labels_path, exist_ok=True)
        counter = 0
        for label_file in labels_list:
            label_path = os.path.join(original_gt_labels, label_file)
            process_label(label_path, new_gt_labels_path, category_mapping)
            counter += 1
        print(f"Processed {counter} labels.")

    original_gt_labels_path = os.path.join(ground_truth_base_dir, "labels")
    print(f"Original gt_labels_path: {original_gt_labels_path}")

    for grouping in groupings:

        grouping = ObjectClass.get_grouping(grouping)
        group_name = grouping["group_name"]
        maps_to_class = int(
            grouping["maps_to"]["class"][0]
        )  # Assuming there's always one target class in the "maps_to"
        print(f"Grouping: {grouping}")
        print(f"Grouping_name: {group_name}")

        # Create a new folder for the output of this grouping
        new_labels_path = os.path.join(ground_truth_base_dir, group_name)
        print(f"Creating new labels folder: {new_labels_path}")
        os.makedirs(new_labels_path, exist_ok=True)

        category_mapping = ObjectClass.get_category_mapping(group_name)
        print(f"Category mapping: {category_mapping}")

        # Choose the specific group you want to process
        process_labels(
            original_gt_labels=original_gt_labels_path,
            new_gt_labels_path=new_labels_path,
            category_mapping=category_mapping,
        )

        # After this, change the ground_truth_base_folder to the new labels folder
        current_ground_truth_base_dir = new_labels_path

        yolo_eval = YoloEvaluator(
            ground_truth_base_folder=current_ground_truth_base_dir,
            predictions_base_folder=predictions_base_dir,
            output_folder=output_dir,
            ground_truth_image_shape=ground_truth_img_shape,
            predictions_image_shape=predictions_img_shape,
            dataset_name=dataset_name,
            model_name=model_name,
            pred_annotations_rel_path=prediction_labels_rel_path,
            splits=splits,
            target_classes=[
                maps_to_class
            ],  # Only take the target class the grouping maps to
            sensitive_classes=sensitive_classes,
            target_classes_conf=target_classes_conf,
            sensitive_classes_conf=sensitive_classes_conf,
        )

        # Total Blurred Area evaluation
        if len(sensitive_classes) > 0:
            tba_results = yolo_eval.evaluate_tba_bias_analysis(grouping=grouping)
            print(f"\n\nAfter TBA: {tba_results}\n\n")
            yolo_eval.save_tba_results_to_csv(results=tba_results, use_groupings=True)
            # Plot precision/recall curves
            if eval_settings["plot_pr_curves"]:
                yolo_eval.plot_tba_pr_f_curves(show_plot=False)
