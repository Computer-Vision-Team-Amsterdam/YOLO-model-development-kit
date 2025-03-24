import glob
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


def process_labels(
    original_gt_labels, ground_truth_rel_path, new_gt_labels_path, category_mapping
):
    """
    Take original ground truth labels, process them according to the mapping
    and write them in a new label folder.
    """
    labels_list = os.listdir(original_gt_labels)
    logger.info(f"There are {len(labels_list)} to be processed.")
    new_gt_labels_path = os.path.join(new_gt_labels_path, ground_truth_rel_path)
    os.makedirs(new_gt_labels_path, exist_ok=True)
    counter = 0
    for label_file in labels_list:
        label_path = os.path.join(original_gt_labels, label_file)
        _process_label(label_path, new_gt_labels_path, category_mapping)
        counter += 1
    logger.info(f"Processed {counter} labels.")


def log_label_counts(labels_folder: str, logger, valid_category_ids: set):
    """
    Counts and logs the number of lines in each label file that start with one of the valid category IDs.

    Parameters
    ----------
    labels_folder : str
        Path to the folder containing label text files.
    logger : logging.Logger
        Logger for logging the counts.
    valid_category_ids : set
        Set of category IDs (integers) that belong to the current grouping.
    """
    counts = {cat_id: 0 for cat_id in valid_category_ids}
    pattern = os.path.join(labels_folder, "**", "*.txt")
    for label_file in glob.glob(pattern, recursive=True):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cat_id = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                if cat_id in valid_category_ids:
                    counts[cat_id] += 1
    logger.info(f"Sample counts in '{labels_folder}': {counts}")
