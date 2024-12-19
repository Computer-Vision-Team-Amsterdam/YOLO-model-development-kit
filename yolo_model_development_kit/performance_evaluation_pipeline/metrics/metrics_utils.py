import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class CategoryManager:
    """Dynamic class to represent object categories for evaluation.
    It contains the category name, ID, and bounding box size thresholds.
    The 'small', 'medium', 'large', and 'all' attributes represent
    the bounding box size categories. The bounds of each category are given as
    fraction of the image surface.
    It also contains the mapping between inference and ground truth categories.
    """

    def __init__(self, categories_json_path: str, mappings_json_path: str = None):
        self._categories = self._load_categories(categories_json_path)
        self._groupings = (
            self._load_mapping(mappings_json_path)
            if mappings_json_path is not None
            else {}
        )

    def _load_categories(self, json_path):
        """Load categories and box size thresholds from a JSON file.
        The function assumes the YOLO convention, so categories start at 0.
        """

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"The specified file '{json_path}' was not found.")
        with open(json_path, "r") as f:
            try:
                categories = json.load(f)
                if not categories.get("categories"):
                    raise ValueError(
                        "The categories JSON file is empty or improperly formatted."
                    )
            except json.JSONDecodeError:
                raise ValueError(f"The file '{json_path}' is not a valid JSON file.")

            return {
                cat["id"]: {
                    "name": cat["name"],
                    "thresholds": tuple(cat.get("thresholds", (0.0, 1.0))),
                }
                for cat in categories["categories"]
            }

    def _load_mapping(self, json_path):
        """Load mapping between inference and ground truth categories from a JSON file."""

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"The specified file '{json_path}' was not found.")
        with open(json_path, "r") as f:
            try:
                groupings = json.load(f)
                if not groupings:
                    raise ValueError(
                        "The groupings JSON file is empty or improperly formatted."
                    )
                return groupings
            except json.JSONDecodeError:
                raise ValueError(f"The file '{json_path}' is not a valid JSON file.")

    def to_dict(
        self, cat_id: int, all_only: bool = False
    ) -> Dict[str, Tuple[float, float]]:
        """Get a dictionary representation of the bounding box size categories for a given category ID."""
        details = self._categories.get(cat_id)
        if not details or "thresholds" not in details:
            raise ValueError(f"No thresholds found for category ID {cat_id}")

        thresholds = details["thresholds"]
        small = (0.0, thresholds[0])
        medium = thresholds
        large = (thresholds[1], 1.0)
        all_bounds = (0.0, 1.0)

        if all_only:
            return {"all": all_bounds}
        else:
            return {
                "all": all_bounds,
                "small": small,
                "medium": medium,
                "large": large,
            }

    def get_name(self, cat_id):
        """Get the category name by ID."""
        return self._categories.get(cat_id, {}).get("name", "Unknown")

    def get_id(self, name):
        """Get the category ID by name."""
        for class_id, details in self._categories.items():
            if details.get("name") == name:
                return class_id
        return None

    def get_grouping(self, grouping_key: str) -> Dict[str, Dict[str, Any]]:
        """Get a specific grouping of categories by key."""
        grouping = self._groupings.get(grouping_key)
        if not grouping:
            raise ValueError(f"No grouping found for key '{grouping_key}'")
        return grouping

    def get_category_mapping(self, grouping_key):
        """Get a mapping dictionary that maps original class IDs to new category IDs for a specific grouping."""
        grouping = self.get_grouping(grouping_key)
        category_mapping = {}
        for category, details in grouping["categories"].items():
            category_id = details["category_id"]
            for class_id in details["classes"]:
                category_mapping[class_id] = category_id
        return category_mapping

    def get_thresholds(self, cat_id):
        """Get the bounding box size thresholds for a given category ID."""
        details = self._categories.get(cat_id)
        if details and "thresholds" in details:
            return details["thresholds"]
        return (0.0, 1.0)  # Default to the whole image

    def all_ids(self):
        """Return all category IDs."""
        return list(self._categories.keys())

    def all_names(self):
        """Return all category names."""
        return [cat.name for cat in self._categories.values()]

    def all_thresholds(self):
        """Return all category thresholds as a dictionary with category IDs as keys."""
        return {
            cat_id: details["thresholds"]
            for cat_id, details in self._categories.items()
            if "thresholds" in details
        }

    def all_groupings(self):
        """Return all groupings of categories."""
        return self._groupings


def parse_labels(
    file_path: str,
) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """
    Parse a YOLO annotation .txt file with the following normalized format:
    `class x_center y_center width height`

    Parameters
    ----------
    file_path: str
        The path to the annotation file to be parsed.

    Returns
    -------
    A tuple: (list of classes, list of (tuples of) bounding boxes)
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    classes = [int(line.strip().split()[0]) for line in lines]
    bounding_boxes = [
        (
            float(line.strip().split()[1]),
            float(line.strip().split()[2]),
            float(line.strip().split()[3]),
            float(line.strip().split()[4]),
        )
        for line in lines
    ]
    return classes, bounding_boxes


def generate_binary_mask(
    bounding_boxes: Union[
        npt.NDArray, List[List[float]], List[Tuple[float, float, float, float]]
    ],
    image_width: int = 3840,
    image_height: int = 2160,
    consider_upper_half: bool = False,
) -> npt.NDArray:
    """
    Create a binary mask where all points inside the given bounding boxes are 1,
    and 0 otherwise.

    Parameters
    ----------
    bounding_boxes:
        Bounding boxes coordinates, either as ndarray of shape(n_boxes, 4),
        as list of lists, or list of tuples.
    image_width: int = 3840
        Width of the image, in pixels.
    image_height: int = 2160
        Height of the image, in pixels.
    consider_upper_half: bool = False
        Only look at the upper half of the bounding boxes (useful for the person
        object class where you want to make sure the head is detected).

    Returns
    -------
    The binary mask.
    """

    mask = np.zeros((image_height, image_width), dtype=bool)

    if len(bounding_boxes):
        bounding_boxes = np.array(bounding_boxes)
        y_min = (
            (bounding_boxes[:, 1] - bounding_boxes[:, 3] / 2) * image_height
        ).astype(int)
        x_min = (
            (bounding_boxes[:, 0] - bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        x_max = (
            (bounding_boxes[:, 0] + bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        if consider_upper_half:
            y_max = (bounding_boxes[:, 1] * image_height).astype(int)
        else:
            y_max = (
                (bounding_boxes[:, 1] + bounding_boxes[:, 3] / 2) * image_height
            ).astype(int)
        for i in range(len(x_min)):
            mask[y_min[i] : y_max[i], x_min[i] : x_max[i]] = 1

    return mask


def compute_fb_score(
    precision: Union[float, npt.NDArray, pd.Series],
    recall: Union[float, npt.NDArray, pd.Series],
    beta: float = 1.0,
    decimals: int = 3,
):
    """
    Compute [F-scores](https://en.wikipedia.org/wiki/F-score) for a pair of
    precision and recall values. The parameter beta determines the relative
    importance of recall w.r.t. precision, i.e., recall is considered _beta_
    times as important as precision.

    Parameters
    ----------
    precision: Union[float, npt.NDArray, pd.Series]
        (List of) precision value(s).
    recall: Union[float, npt.NDArray, pd.Series]
        (List of) recall value(s).
    beta: float = 1.0
        Relative importance of recall w.r.t. precision.
    decimals: int = 3
        Rounds f-score to the given number of decimals.

    Returns
    -------
    F-scores for the precision and recall pair(s).
    """
    if isinstance(precision, pd.Series):
        precision = precision.to_numpy()
    if isinstance(recall, pd.Series):
        recall = recall.to_numpy()
    return np.round(
        (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall),
        decimals=decimals,
    )
