import numpy as np
from ultralytics.engine.results import Boxes

from yolo_model_development_kit.inference_pipeline.source.model_result import (
    ModelResult,
)
from yolo_model_development_kit.inference_pipeline.source.YOLO_inference import (
    YOLOInference,
)


def test_sahi_to_yolo_results_conversion(
    sahi_results_mock,
    expected_boxes,
    expected_confs,
    expected_classes,
    description="Test",
):
    image_height, image_width = 720, 1280
    mock_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    mock_path = "mock_image.jpg"
    mock_category_mapping = {0: "person", 1: "license plate"}
    mock_speed = {"slice": 10.0, "prediction": 50.0}

    print(f"Running: {description}")

    sahi_results = YOLOInference.process_sahi_results_to_yolo_results(
        sahi_results=sahi_results_mock,
        image=mock_image,
        image_path=mock_path,
        category_mapping=mock_category_mapping,
        speed=mock_speed,
    )

    target_classes = [0, 1]
    sensitive_classes = [1]

    model_result = ModelResult(
        model_result=sahi_results,
        target_classes=target_classes,
        sensitive_classes=sensitive_classes,
        target_classes_conf=0.5,
        sensitive_classes_conf=0.7,
        save_image=False,
        save_labels=False,
        save_all_images=False,
    )

    assert model_result.result.orig_shape == (
        image_height,
        image_width,
    ), "Original shape mismatch!"
    assert isinstance(model_result.result.boxes, Boxes), "Boxes type mismatch!"
    assert model_result.result.path == mock_path, "Image path mismatch!"
    assert (
        model_result.result.boxes.xyxy.shape[0] == expected_boxes
    ), f"Incorrect number of boxes! Expected {expected_boxes}."
    assert (
        model_result.result.boxes.conf.tolist() == expected_confs
    ), "Confidence values mismatch!"
    assert (
        model_result.result.boxes.cls.tolist() == expected_classes
    ), "Class IDs mismatch!"
    assert model_result.result.names == mock_category_mapping, "Class names mismatch!"
    assert model_result.result.speed == mock_speed, "Speed values mismatch!"

    print("Test passed!")


if __name__ == "__main__":
    sahi_results_mock_with_detections = [
        {
            "image_id": None,
            "bbox": [290.86, 1684.02, 32.24, 75.63],
            "score": 0.8050000071525574,
            "category_id": 0,
            "category_name": "person",
            "segmentation": [],
            "iscrowd": 0,
            "area": 2438,
        },
        {
            "image_id": None,
            "bbox": [960.08, 1990.67, 95.77, 49.07],
            "score": 0.7940000295639038,
            "category_id": 1,
            "category_name": "license plate",
            "segmentation": [],
            "iscrowd": 0,
            "area": 4699,
        },
    ]
    test_sahi_to_yolo_results_conversion(
        sahi_results_mock_with_detections,
        expected_boxes=2,
        expected_confs=[0.8050000071525574, 0.7940000295639038],
        expected_classes=[0, 1],
        description="Test with detections",
    )

    sahi_results_mock_no_detections = []
    test_sahi_to_yolo_results_conversion(
        sahi_results_mock_no_detections,
        expected_boxes=0,
        expected_confs=[],
        expected_classes=[],
        description="Test with no detections",
    )
