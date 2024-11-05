import json
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class ObjectClass:
    """Dynamic class to represent object categories for evaluation."""

    _categories = {}

    @classmethod
    def load_categories(cls, json_path):
        """Load categories from a JSON file."""
        with open(json_path, "r") as file:
            categories = json.load(file)
            cls._categories = {
                cat["id"]: cat["name"] for cat in categories["categories"]
            }

    @classmethod
    def get_name(cls, cat_id):
        """Get the category name by ID."""
        return cls._categories.get(cat_id, "Unknown")

    @classmethod
    def get_id(cls, name):
        for class_id, class_name in cls._categories.items():
            if class_name == name:
                return class_id
        return None

    @classmethod
    def all_ids(cls):
        """Return all category IDs."""
        return list(cls._categories.keys())

    @classmethod
    def all_names(cls):
        """Return all category names."""
        return list(cls._categories.values())


class BoxSize:
    """
    This class is used to represent bounding box size categories 'small',
    'medium', 'large', and 'all'. The bounds of each category are given as
    fraction of the image surface. They are dynamically loaded from a JSON file.

    Parameters
    ----------
    bounds: Tuple[float, float]
        The two relevant bounds between small and medium, and medium and large.
    """

    all: Tuple[float, float] = (0.0, 1.0)
    small: Tuple[float, float]
    medium: Tuple[float, float]
    large: Tuple[float, float]
    thresholds: Dict[str, Tuple[float, float]] = {}

    def __init__(self, bounds: Tuple[float, float]):
        self.small = (0.0, bounds[0])
        self.medium = bounds
        self.large = (bounds[1], 1.0)

    @classmethod
    def load_thresholds(cls, file_path: str) -> None:
        """Load bounding box thresholds from a JSON file."""
        with open(file_path, "r") as f:
            cls.thresholds = {
                name: tuple(bounds) for name, bounds in json.load(f).items()
            }

    @classmethod
    def from_objectclass(cls, object_class_name: str):
        """
        Create a BoxSize object based on the object's name. This will return a
        BoxSize instance with bounds set to the appropriate values for that
        object name.

        Parameters
        ----------
        object_class_name: str
            The name of the object to get the BoxSize for. e.g. `BoxSize.from_objectclass(ObjectClass.get_name(target_class))`.

        Returns
        -------
        BoxSize instance with the appropriate bounds.
        """
        bounds = cls.thresholds.get(object_class_name)
        if not bounds:
            raise ValueError(
                f"No size bounds found for class '{object_class_name}' in the thresholds."
            )
        return cls(bounds)

    def to_dict(self, all_only: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        Get a dict representation of this instance.

        Parameters
        ----------
        all_only: bool = False
            Whether or not to only return the bounds for 'all'. This is purely a
            convenience method for the 'single_size_only' case of several
            metrics and serves no other practical purpose.

        Returns
        -------
        A dictionary with the size categories as keys and their bounds as
        values.
        """
        if all_only:
            return {"all": self.all}
        else:
            return {
                "all": self.all,
                "small": self.small,
                "medium": self.medium,
                "large": self.large,
            }

    def __repr__(self) -> str:
        return repr(self.medium)


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
