import numpy as np
from ultralytics.engine.results import Boxes

from yolo_model_development_kit.inference_pipeline.source.model_result import (
    ModelResult,
)
from yolo_model_development_kit.inference_pipeline.source.YOLO_inference import (
    YOLOInference,
)


def test_model_result_with_sahi_results():
    image_height, image_width = 720, 1280
    mock_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    mock_path = "mock_image.jpg"
    mock_category_mapping = {0: "person", 1: "license plate"}
    mock_speed = {"slice": 10.0, "prediction": 50.0}

    sahi_results_mock = [
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
    assert model_result.result.boxes.xyxy.shape[0] == 2, "Incorrect number of boxes!"
    assert model_result.result.boxes.conf.tolist() == [
        0.8050000071525574,
        0.7940000295639038,
    ], "Confidence values mismatch!"
    assert model_result.result.boxes.cls.tolist() == [0, 1], "Class IDs mismatch!"
    assert model_result.result.names == mock_category_mapping, "Class names mismatch!"
    assert model_result.result.speed == mock_speed, "Speed values mismatch!"

    print("Test passed: ModelResult works correctly with Results.")


if __name__ == "__main__":
    test_model_result_with_sahi_results()
