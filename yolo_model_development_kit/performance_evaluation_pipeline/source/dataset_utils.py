import logging
import os

logger = logging.getLogger(__name__)


def _process_label(label_path, new_labels_path, category_mapping):
    """Maps category IDs to original ground truth labels based on the grouping specified"""
    with open(label_path, "r") as label_file:
        lines = label_file.readlines()

    new_lines = []
    for line in lines:
        label_fields = line.strip().split()
        original_class_id = int(label_fields[0])

        if original_class_id in category_mapping:
            new_class_id = category_mapping[original_class_id]
            label_fields[0] = str(new_class_id)
        else:
            new_class_id = original_class_id

        new_lines.append(" ".join(label_fields))

    new_label_path = os.path.join(new_labels_path, os.path.basename(label_path))
    with open(new_label_path, "w") as label_file:
        label_file.write("\n".join(new_lines))


def process_labels(original_gt_labels, new_gt_labels_path, category_mapping):
    """
    Take original ground truth labels, process them according to the mapping
    and write them in a new label folder.
    """
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
