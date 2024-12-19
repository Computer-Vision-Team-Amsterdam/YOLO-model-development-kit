import json
import logging
from typing import Any, Dict, List, Tuple, Union

from pycocotools.coco import COCO

from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (
    CategoryManager,
    CustomCOCOeval,
)

logger = logging.getLogger("performance_evaluation")


def execute_custom_coco_eval(
    coco_ground_truth_json: str,
    coco_predictions_json: str,
    category_manager: CategoryManager,
    predicted_img_shape: Tuple[int, int],
    classes: List[int] = None,
    print_summary: bool = False,
    precision: int = 3,
) -> Dict[str, float]:
    """
    Runs our custom COCO evaluation on a ground truth dataset and YOLO model
    predictions.

    Parameters
    ----------
    coco_ground_truth_json: str
        Path to JSON file with ground truth annotations in the COCO format.
    coco_predictions_json: str
        Path to JSON file with prediction annotations in the COCO format.
    category_manager: CategoryManager
        CategoryManager object containing the object classes to evaluate.
    predicted_img_shape: Tuple[int, int]
        Shape of images in the coco_predictions_json. Should equal the shape of
        ground truth images. Used as a sanity check.
    classes: List[int] = None
        Which classes to evaluate (default is all).
    print_summary: bool = False
        Whether or not to have CustomCOCOeval print a summary of results.
    precision: int = 3
        Rounds precision, recall, and f1_score to the given number of decimals.

    Returns
    -------
    Dictionary with the results:

            {
            [object_class]: {
                "AP@50-95_all": float,
                "AP@75_all": float,
                "AP@50_all": float,
                "AP@50_small": float,
                "AP@50_medium": float,
                "AP@50_large": float,
                "AR@50-95_all": float,
                "AR@75_all": float,
                "AR@50_all": float,
                "AR@50_small": float,
                "AR@50_medium": float,
                "AR@50_large": float,
            }
        }
    """
    COCO_gt = COCO(coco_ground_truth_json)  # init annotations api
    try:
        COCO_dt = COCO_gt.loadRes(coco_predictions_json)  # init predictions api
    except FileNotFoundError:
        raise Exception(
            f"The specified file '{coco_predictions_json}' was not found."
            f"The file is created at the above path if you run yolo validation with"
            f"the --save-json flag enabled."
        )
    evaluation = CustomCOCOeval(COCO_gt, COCO_dt, "bbox")

    with open(coco_ground_truth_json) as f:
        data = json.load(f)

    height = data["images"][0]["height"]
    width = data["images"][0]["width"]
    if width != predicted_img_shape[0] or height != predicted_img_shape[1]:
        logger.warning(
            f"You're trying to run evaluation on images of size {width} x {height}, "
            "but the coco annotations have been generated from images of size "
            f"{predicted_img_shape[0]} x {predicted_img_shape[1]}."
            "Why is it a problem? Because the coco annotations that the metadata produces and the "
            " *_predictions.json produced by the yolo run are both in absolute format,"
            "so we must compare use the same image sizes."
            "Solutions: 1. Use images for validation that are the same size as the ones you used for the "
            "annotations. 2. Re-compute the coco_annotations_json using the right image shape."
        )

    # Set evaluation params
    image_names = [image["id"] for image in data["images"]]
    evaluation.params.imgIds = image_names  # image IDs to evaluate
    # If no specific classes are passed, use all available class IDs.
    if classes is None:
        classes = category_manager.all_ids()
    evaluation.params.catIds = list(classes)
    class_labels = [category_manager.get_name(cat_id) for cat_id in classes]
    evaluation.params.catLbls = class_labels

    # We need to overwrite the default area ranges for the bounding box size differentiation
    img_area = height * width
    areaRng: List[Dict[Union[str, int], Any]] = []
    for areaRngLbl in evaluation.params.areaRngLbl:
        aRng: Dict[Union[str, int], Tuple[float, float]] = {"areaRngLbl": areaRngLbl}
        for class_id in classes:
            # Dynamically retrieve the size range for each category.
            box_size = category_manager.to_dict(class_id)
            box_range = box_size[areaRngLbl]
            aRng[class_id] = (box_range[0] * img_area, box_range[1] * img_area)
        areaRng.append(aRng)
    evaluation.params.areaRng = areaRng

    # Run the evaluation
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize(print_summary=print_summary)

    return _stats_to_dict(evaluation, precision)


def _stats_to_dict(eval: CustomCOCOeval, precision: int) -> Dict:
    """Convert CustomCOCOeval results to dict."""
    keys = [
        "AP@50-95_all",
        "AP@75_all",
        "AP@50_all",
        "AP@50_small",
        "AP@50_medium",
        "AP@50_large",
        "AR@50-95_all",
        "AR@75_all",
        "AR@50_all",
        "AR@50_small",
        "AR@50_medium",
        "AR@50_large",
    ]
    values = eval.stats
    return {
        key: round(value, precision) if value else None
        for key, value in zip(keys, values)
    }
