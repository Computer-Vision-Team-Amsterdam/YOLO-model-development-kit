import logging
import os

logger = logging.getLogger(__name__)


def _process_label(label_path, new_labels_path, category_mapping):
    """Maps category IDs to original ground truth labels based on the grouping specified"""
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
    Takes mapped ground truth labels and make a new labels folder
    """
    # List all labels (.txt files) in the dataset labels folder
    labels_list = os.listdir(original_gt_labels)
    logger.info(f"There are {len(labels_list)} to be processed.")
    new_gt_labels_path = os.path.join(new_gt_labels_path, "labels")
    os.makedirs(new_gt_labels_path, exist_ok=True)
    counter = 0
    for label_file in labels_list:
        label_path = os.path.join(original_gt_labels, label_file)
        _process_label(label_path, new_gt_labels_path, category_mapping)
        counter += 1
    logger.info(f"Processed {counter} labels.")
