import logging
import os
from types import NoneType
from typing import Any, Dict, List, Union

import numpy as np
import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from ultralytics.engine.results import Results

from yolo_model_development_kit.inference_pipeline.source.input_image import InputImage
from yolo_model_development_kit.inference_pipeline.source.model_result import (
    ModelResult,
)

logger = logging.getLogger("inference_pipeline")


class YOLOInference:
    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
    ) -> None:
        """
        This class runs inference on images using a pre-trained YOLO model to
        detect a set of target classes. Optionally, sensitive classes are
        blurred and output images are stored.

        Input images will be de-fisheyed and resized if needed. When one or more
        objects belonging to one of the target classes are detected in an image,
        the bounding boxes for those detections are stored in a .txt file with
        the same name as the image. If save_images or save_all_images is set,
        the output will be saved as an image with the original file name, with
        sensitive classes blurred, and bounding boxes of target classes drawn.

        The difference between save_image and save_all_images is that the former
        will only save images when an object belonging to one of the
        target_classes is detected in the image.

        Parameters
        ----------
        images_folder: str
            Location of images to run inference on. If the location contains
            sub-folders, this folder structure will be preserved in the output.
        output_folder: str
            Location where output (annotation labels and possible images) will
            be stored.
        model_path: str
            Location of the pre-trained YOLO model.
        inference_settings: Dict
            Settings for the model, which contains:
                model_params: Dict
                    Inference parameters for the YOLO model:
                        img_size, save_img_flag, save_txt_flag, save_conf_flag, conf
                target_classes: List
                    List of target classes for which bounding boxes will be predicted.
                sensitive_classes: List
                    List of sensitive classes which will be blurred in output images.
                target_classes_conf: Optional[float] = None
                    Optional: confidence threshold for target classes. Only detections
                    above this threshold will be considered. If omitted,
                    inference_param["conf"] will be used.
                sensitive_classes_conf: Optional[float] = None
                    Optional: confidence threshold for sensitive classes. Only
                    detections above this threshold will be considered. If omitted,
                    inference_param["conf"] will be used.
                output_image_size: Optional[Tuple[int, int]] = None
                    Optional: output images will be resized to these (width, height)
                    dimensions if set.
                defisheye_flag: bool = False
                    Whether or not to apply distortion correction to the input images.
                defisheye_params: Dict = {}
                    If defisheye_flag is True, these distortion correction parameters
                    will be used. Contains "camera_matrix", "distortion_params", and
                    "input_image_size" (size of images used to compute these
                    parameters).
                save_images: bool = False
                    Whether or not to save the output images.
                save_labels: bool = True
                    Whether or not to save the annotation labels.
                save_all_images: bool = False
                    Whether to save all processed images (TRue) or only those containing
                    objects belonging to one of the target classes (False).
                save_images_subfolder: Optional[str] = None
                    Optional: sub-folder in which to store output images.
                save_labels_subfolder: Optional[str] = None
                    Optional: sub-folder in which to store annotation labels.
                batch_size: int = 1
                    Batch size for inference.
        """
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.use_sahi = inference_settings["use_sahi"]
        self.sahi = inference_settings["sahi_params"]

        self.inference_params = {
            "imgsz": inference_settings["model_params"].get("img_size", 640),
            "save": inference_settings["model_params"].get("save_img_flag", False),
            "save_txt": inference_settings["model_params"].get("save_txt_flag", False),
            "save_conf": inference_settings["model_params"].get(
                "save_conf_flag", False
            ),
            "conf": inference_settings["model_params"].get("conf", 0.25),
            "project": output_folder,
        }

        logger.debug(f"Inference_params: {self.inference_params}")
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Output folder: {self.output_folder}")

        if self.use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type=self.sahi["model_type"],
                model_path=model_path,
                confidence_threshold=self.inference_params["conf"],
            )
            logger.info(f"Using SAHI model with params: {self.sahi}.")
        else:
            self.model = YOLO(model=model_path, task="detect")
            logger.info("Using YOLO model.")

        self.target_classes = inference_settings["target_classes"]
        self.sensitive_classes = inference_settings["sensitive_classes"]
        self.target_classes_conf = (
            inference_settings["target_classes_conf"]
            if inference_settings["target_classes_conf"]
            else self.inference_params["conf"]
        )
        self.sensitive_classes_conf = (
            inference_settings["sensitive_classes_conf"]
            if inference_settings["sensitive_classes_conf"]
            else self.inference_params["conf"]
        )
        logger.debug(
            f"Using confidence thresholds: target_classes: {self.target_classes_conf}, "
            f"sensitive_classes: {self.sensitive_classes_conf}"
        )
        self.output_image_size = inference_settings["output_image_size"]
        self.save_detections = inference_settings["save_detection_images"]
        self.save_labels = inference_settings["save_detection_labels"]
        self.save_all_images = inference_settings["save_all_images"]
        self.detections_subfolder = (
            inference_settings["outputs"]["detections_subfolder"]
            if inference_settings["outputs"]["detections_subfolder"]
            else ""
        )
        self.labels_subfolder = (
            inference_settings["outputs"]["labels_subfolder"]
            if inference_settings["outputs"]["labels_subfolder"]
            else ""
        )
        self.batch_size = inference_settings["model_params"]["batch_size"]
        self.folders_and_frames: Dict[str, List[str]] = {}

    def run_pipeline(self) -> None:
        """
        Runs the inference pipeline:
        - find the images to detect;
        - detects everything;
        - stores labels if required;
        - stores images if required, with
            - sensitive classes blurred;
            - target classes bounding boxes drawn;
        """
        logger.info(f"Running detection pipeline on {self.images_folder}..")
        if not self.folders_and_frames:  # type: ignore
            self.folders_and_frames = self._find_image_paths_and_group_by_folder(
                root_folder=self.images_folder
            )
        logger.info(
            f"Total number of images: {sum(len(frames) for frames in self.folders_and_frames.values())}"
        )
        self._process_batches()

    def _load_image(
        self, image_path: Union[os.PathLike, str], child_class=InputImage
    ) -> Union[InputImage, None]:
        """
        This method can be extended in the child class to perform pre-processing
        of the images before the inference. To do this we can also extend
        InputImage and add the extra functions are needed, for example a
        function to defisheye the image. An example of this can be found in the
        Objectherkenning-Openbare-Ruimte repo.

        Parameters
        ----------
        image_path : Union[os.PathLike, str]
            path where the image is located
        child_class : _type_, optional
            Type of object to create, by default InputImage but it can also be a
            child of InputImage

        Returns
        -------
        InputImage
            the preprocessed image ready to be inferenced, or None if the image
            could not be loaded
        """
        image = child_class(image_full_path=str(image_path))

        if isinstance(image.image, NoneType):
            logger.warning(f"Image could not be loaded: {image_path}")
            return None

        if self.output_image_size:
            image.resize(output_image_size=self.output_image_size)

        return image

    def _run_sahi_inference(
        self, batch_images: List[InputImage], batch_image_paths: List[str]
    ) -> List[Results]:
        """
        Run inference using SAHI for a batch of images.

        Parameters
        ----------
        batch_images: List[InputImage]
            List of images in the current batch.
        batch_image_paths: List[str]
            List of images paths in the current batch.

        Returns
        -------
        List[Results]
            List of Results objects for the batch.
        """
        batch_results = []
        for image, image_path in zip(batch_images, batch_image_paths):
            result = get_sliced_prediction(
                image=image,
                detection_model=self.sahi_model,
                slice_height=self.sahi["slice_height"],
                slice_width=self.sahi["slice_width"],
                overlap_height_ratio=self.sahi["overlap_height_ratio"],
                overlap_width_ratio=self.sahi["overlap_width_ratio"],
            )
            object_prediction_list = result.to_coco_annotations()
            category_mapping = {
                prediction.category.id: prediction.category.name
                for prediction in result.object_prediction_list
            }

            sahi_result = self.process_sahi_results_to_yolo_results(
                sahi_results=object_prediction_list,
                image=image,
                image_path=image_path,
                category_mapping=category_mapping,
                speed=result.durations_in_seconds,
            )
            batch_results.append(sahi_result)
        return batch_results

    def _run_yolo_inference(
        self, batch_images: List[InputImage], folder_name: str
    ) -> List[Results]:
        """
        Run inference using YOLO for a batch of images.

        Parameters
        ----------
        batch_images: List[InputImage]
            List of images in the current batch.
        folder_name: str
            Name of the folder containing the batch images.

        Returns
        -------
        List[Results]
            List of Results objects for the batch.
        """
        self.inference_params["source"] = batch_images
        self.inference_params["name"] = folder_name
        return self.model(**self.inference_params)

    def _process_batches(self) -> None:
        """
        Process all images in all sub-folders in batches of size batch_size.
        """
        for folder_name, images in self.folders_and_frames.items():
            logger.debug(
                f"Running inference on folder: {os.path.relpath(folder_name, self.images_folder)}"
            )
            image_paths = [os.path.join(folder_name, image) for image in images]
            logger.debug(f"Number of images to detect: {len(image_paths)}")
            processed_images = 0
            for i in range(0, len(image_paths), self.batch_size):
                batch_image_paths = image_paths[i : i + self.batch_size]
                batch_input_images = [
                    self._load_image(image_path) for image_path in batch_image_paths
                ]

                # Check if all images could be loaded
                correct_idx = [
                    i for i, img in enumerate(batch_input_images) if img is not None
                ]
                # Prepare correctly loaded images
                batch_images = [batch_input_images[idx].image for idx in correct_idx]
                batch_image_paths = [batch_image_paths[idx] for idx in correct_idx]
                if len(batch_images) < 1:
                    continue

                if self.use_sahi:
                    batch_results = self._run_sahi_inference(
                        batch_images, batch_image_paths
                    )
                else:
                    batch_results = self._run_yolo_inference(batch_images, folder_name)

                self._process_detections(
                    model_results=batch_results,
                    image_paths=batch_image_paths,
                )
                processed_images += len(batch_image_paths)

            logger.debug(f"Number of images processed: {processed_images}")

    def _process_detections(
        self, model_results: List[Results], image_paths: List[str]
    ) -> None:
        """
        Process the YOLO inference Results objects:
        - save output image
        - save annotation labels

        Parameters
        ----------
        model_results: List[Results]
            List of YOLO inference Results objects.
        image_paths: List[str]
            List of input image paths corresponding to the Results.
        """
        for result, image_path in zip(model_results, image_paths):
            model_result = ModelResult(
                model_result=result,
                target_classes=self.target_classes,
                sensitive_classes=self.sensitive_classes,
                target_classes_conf=self.target_classes_conf,
                sensitive_classes_conf=self.sensitive_classes_conf,
                save_image=self.save_detections,
                save_labels=self.save_labels,
                save_all_images=self.save_all_images,
            )

            # Get the relative path of the image w.r.t. the input folder. This
            # is used to preserve the folder structure in the output.
            base_folder = os.path.dirname(
                os.path.relpath(image_path, self.images_folder)
            )
            output_base_path = os.path.join(self.output_folder, base_folder)
            image_output_path = os.path.join(
                output_base_path, self.detections_subfolder
            )
            labels_output_path = os.path.join(output_base_path, self.labels_subfolder)
            image_file_name = os.path.basename(image_path)

            logger.debug(f"Processing image: {image_file_name}")

            model_result.process_detections_and_blur_sensitive_data(
                output_folder=image_output_path,
                image_file_name=image_file_name,
                labels_output_folder=labels_output_path,
            )

    @staticmethod
    def _find_image_paths_and_group_by_folder(root_folder: str) -> Dict[str, List[str]]:
        """
        Find all image files in the root_folder, group them by sub-folder, and
        return these as a dictionary mapping.

        Parameters
        ----------
        root_folder: str
            The root folder.

        Returns
        -------
        A dictionary mapping sub-folders to images contained in them:
        `{"folder_name": ["img_1.jpg", "img_2.jpg", ...]}`
        """
        folders_and_frames: Dict[str, List[str]] = {}
        for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    if foldername not in folders_and_frames:
                        folders_and_frames[foldername] = [filename]
                    else:
                        folders_and_frames[foldername].append(filename)
        return folders_and_frames

    @staticmethod
    def process_sahi_results_to_yolo_results(
        sahi_results: List[Dict[str, Any]],
        image: np.ndarray,
        image_path: str,
        category_mapping: Dict[str, str],
        speed: Dict[str, float],
    ) -> Results:
        """
        Converts SAHI results into a YOLO-style Results object.

        Args:
            sahi_results (List[Dict[str, Any]]): SAHI COCO-style annotations.
            image (np.ndarray): Original image array.
            image_path (str): Path to the original image.

        Returns:
            Results: A Results object with SAHI detection boxes
        """

        boxes_data = []
        for result in sahi_results:
            x_min, y_min, bbox_width, bbox_height = result["bbox"]
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height

            boxes_data.append(
                [x_min, y_min, x_max, y_max, result["score"], result["category_id"]]
            )

        if len(boxes_data) == 0:
            boxes_tensor = torch.empty(
                (0, 6), dtype=torch.float32
            )  # Boxes class expects tensor with shape (0,6)
        else:
            boxes_tensor = torch.tensor(np.array(boxes_data), dtype=torch.float32)

        names = {int(k): v for k, v in category_mapping.items()}

        return Results(
            boxes=boxes_tensor,
            orig_img=image,
            path=image_path,
            names=names,
            speed=speed,
        )
