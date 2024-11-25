import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class ObjectClass:
    """Dynamic class to represent object categories for evaluation.
    It contains the category name, ID, and bounding box size thresholds.
    The 'small', 'medium', 'large', and 'all' attributes represent
    the bounding box size categories. The bounds of each category are given as
    fraction of the image surface.
    """

    _categories = {}
    _grouped_categories_by_type = {}

    @classmethod
    def load_categories(cls, json_path):
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

            cls._categories = {
                cat["id"]: {
                    "name": cat["name"],
                    "thresholds": tuple(
                        cat.get("thresholds", (0.0, 1.0))
                    ),  # Default if 'thresholds' is missing
                }
                for cat in categories["categories"]
            }

    @classmethod
    def to_dict(
        cls, cat_id: int, all_only: bool = False
    ) -> Dict[str, Tuple[float, float]]:
        """Get a dictionary representation of the bounding box size categories for a given category ID."""
        details = cls._categories.get(cat_id)
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

    @classmethod
    def reset_categories(cls):
        """Reset _categories to an empty dictionary."""
        cls._categories = {}

    @classmethod
    def load_categories_from_dict(cls, category_dict):
        """Load categories directly from a dictionary.

        Parameters
        ----------
        category_dict: dict
            The dictionary should contain categories as values, not as keys.
        """
        # Iterate over each group in the provided dictionary and load categories
        cls.reset_categories()  # Clear existing categories before loading new ones
        for group_name, category_id in category_dict.items():
            if category_id in cls._categories:
                raise ValueError(
                    f"Category ID {category_id} already exists in the loaded categories."
                )

            cls._categories[category_id] = {
                "name": group_name,
                "thresholds": (0.0, 1.0),  # Set default thresholds if needed
            }

    @classmethod
    def apply_groupings(cls, grouping_json_path, group_types):
        """Apply multiple groupings based on the JSON file.

        Parameters
        ----------
        grouping_json_path: str
            Path to the grouping JSON file.
        group_types: list
            List of group types to apply.

        Example JSON file:
        {
            "group_by_X": {
                "group_id": 1,
                "group_name": "group_1",
                "categories": {
                    "category_1": {
                        "category_id": 101,
                        "classes": [1, 2, 3]
                    },
                    "category_2": {
                        "category_id": 102,
                        "classes": [4, 5, 6]
                    }
                }
            },
            "group_by_Y": {
                ...
            }
        }
        """
        if not os.path.exists(grouping_json_path):
            raise FileNotFoundError(
                f"The specified file '{grouping_json_path}' was not found."
            )
        with open(grouping_json_path, "r") as f:
            try:
                grouping_data = json.load(f)
                if len(grouping_data) == 0:
                    raise ValueError(
                        "The grouping JSON file is empty or improperly formatted."
                    )
            except json.JSONDecodeError:
                raise ValueError(
                    f"The file '{grouping_json_path}' is not a valid JSON file."
                )

            # Apply each grouping type from the list
            for group_type in group_types:
                if group_type not in grouping_data:
                    raise ValueError(
                        f"Group type '{group_type}' not found in grouping file."
                    )

                # Load the specified group type under the main group
                group_data = grouping_data[group_type]
                grouped_categories = {
                    group_name: group_details["category_id"]
                    for group_name, group_details in group_data["categories"].items()
                }

                cls._grouped_categories_by_type[group_type] = grouped_categories

    @classmethod
    def get_group(cls, group_type):
        """Return a dictionary of categories for a specific group type."""
        if group_type in cls._grouped_categories_by_type:
            return cls._grouped_categories_by_type[group_type]
        else:
            raise ValueError(
                f"Group type '{group_type}' not found in loaded groupings."
            )

    @classmethod
    def get_name(cls, cat_id):
        """Get the category name by ID."""
        return cls._categories.get(cat_id, {}).get("name", "Unknown")

    @classmethod
    def get_id(cls, name):
        """Get the category ID by name."""
        for class_id, details in cls._categories.items():
            if details.get("name") == name:
                return class_id
        return None

    @classmethod
    def get_thresholds(cls, cat_id):
        """Get the bounding box size thresholds for a given category ID."""
        details = cls._categories.get(cat_id)
        if details and "thresholds" in details:
            return details["thresholds"]
        print(
            f"No thresholds found for category ID {cat_id}. Returning default thresholds."
        )
        return (0.0, 1.0)  # Default to the whole image

    @classmethod
    def all_ids(cls):
        """Return all category IDs."""
        return list(cls._categories.keys())

    @classmethod
    def all_names(cls):
        """Return all category names."""
        return [cat["name"] for cat in cls._categories.values()]

    @classmethod
    def all_thresholds(cls):
        """Return all category thresholds as a dictionary with category IDs as keys."""
        return {
            cat_id: details["thresholds"]
            for cat_id, details in cls._categories.items()
            if "thresholds" in details
        }


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
