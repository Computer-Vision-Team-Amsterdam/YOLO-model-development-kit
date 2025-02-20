# Adapted from: https://www.kaggle.com/code/siddharthkumarsah/convert-yolo-annotations-to-coco-pascal-voc?scriptVersionId=123233495&cellId=1
# TODO: Move this to CVToolkit

import json
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (
    CategoryManager,
)

logger = logging.getLogger(__name__)


def convert_yolo_predictions_to_coco_json(
    predictions_dir: str,
    image_shape: Tuple[int, int],
    labels_rel_path: str = "labels",
    splits: Optional[List[str]] = ["train", "val", "test"],
    output_dir: Optional[str] = None,
    conf: Optional[float] = None,
) -> List[str]:
    if not splits:
        splits = [""]
    if not output_dir:
        output_dir = predictions_dir

    output_files: List[str] = []

    for split in splits:
        label_dir = os.path.join(predictions_dir, labels_rel_path, split)

        prediction_data = _convert_predictions_split(label_dir, image_shape, conf)

        output_file = os.path.join(output_dir, f"coco_predictions_{split}.json")
        with open(output_file, "w") as f:
            f.write(json.dumps(prediction_data))
        output_files.append(output_file)

    return output_files


def _convert_predictions_split(
    label_dir: str,
    image_shape: Tuple[int, int],
    conf: Optional[float] = None,
) -> List[Dict]:
    prediction_data = []
    for pred_file in pathlib.Path(label_dir).glob("*.txt"):
        with open(pred_file) as f:
            for annotation in f.readlines():
                parts = annotation.strip().split()
                cat, xn, yn, wn, hn = map(float, parts[0:5])
                if len(parts) > 5:
                    score = float(parts[5])
                else:
                    score = 1.0
                if conf and (score < conf):
                    continue

                width = wn * image_shape[0]
                height = hn * image_shape[1]
                x = (xn * image_shape[0]) - (width / 2)
                y = (yn * image_shape[1]) - (height / 2)
                prediction_data.append(
                    {
                        "image_id": pred_file.stem,
                        "category_id": int(cat),
                        "bbox": [x, y, width, height],
                        "score": score,
                    }
                )
    return prediction_data


def convert_yolo_dataset_to_coco_json(
    dataset_dir: str,
    category_manager: CategoryManager,
    splits: Optional[List[str]] = ["train", "val", "test"],
    fixed_image_shape: Optional[Tuple[int, int]] = None,
    output_dir: Optional[str] = None,
) -> List[str]:
    if not splits:
        splits = [""]
    if not output_dir:
        output_dir = dataset_dir

    output_files: List[str] = []

    for split in splits:
        image_dir = os.path.join(dataset_dir, "images", split)
        label_dir = os.path.join(dataset_dir, "labels", split)

        coco_dataset = _convert_dataset_split(
            image_dir, label_dir, category_manager, fixed_image_shape
        )

        output_file = os.path.join(output_dir, f"coco_gt_{split}.json")
        with open(output_file, "w") as f:
            json.dump(coco_dataset, f)
        output_files.append(output_file)

    return output_files


def _convert_dataset_split(
    image_dir: str,
    label_dir: str,
    category_manager: CategoryManager,
    fixed_image_shape: Optional[Tuple[int, int]] = None,
) -> Dict:
    image_list: List[Dict] = []
    annotation_list: List[Dict] = []

    categories = [
        {"id": cat_id, "name": category_manager.get_name(cat_id)}
        for cat_id in category_manager.all_ids()
    ]

    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)

        try:
            width, height = Image.open(image_path).size
        except UnidentifiedImageError:
            logger.warning(f"UnidentifiedImageError: {image_path} could not be opened.")
            continue

        if fixed_image_shape is not None:
            fix_width, fix_height = fixed_image_shape
            if abs(fix_width / fix_height - width / height) > 1e-06:
                logger.warning(
                    f"Ratio mismatch: fixed image shape {fixed_image_shape} "
                    f"does not match true image shape {(width, height)} "
                    f"for image {image_file}"
                )
            width = fix_width
            height = fix_height

        image_dict = {
            "id": image_file.split(".")[0],
            "width": width,
            "height": height,
            "file_name": image_file,
        }
        image_list.append(image_dict)

        annotation_file = os.path.join(label_dir, f'{image_file.split(".")[0]}.txt')
        if not os.path.isfile(annotation_file):
            continue

        with open(annotation_file) as f:
            annotations = f.readlines()

            for ann in annotations:
                cl, x, y, w, h = map(float, ann.strip().split()[0:5])
                x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                ann_dict = {
                    "id": len(annotation_list),
                    "image_id": image_file.split(".")[0],
                    "category_id": int(cl),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                }
                annotation_list.append(ann_dict)

    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": image_list,
        "annotations": annotation_list,
    }

    return coco_dataset
