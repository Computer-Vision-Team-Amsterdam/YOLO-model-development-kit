import logging
from typing import Tuple

import cv2

logger = logging.getLogger("inference_pipeline")


class InputImage:
    def __init__(self, image_full_path: str) -> None:
        """
        This class is used to load an input image.
        In case of more needed it can be extended to add for example resizing or blurring.

        Parameters
        ----------
        image_full_path: str
            Path to the input image.
        """
        self.image = cv2.imread(str(image_full_path))

    def resize(self, output_image_size: Tuple[int, int]) -> None:
        """
        Resize the image if needed.

        Parameters
        ----------
        output_image_size: Tuple[int, int]
            Output size as Tuple `(width, height)`.
        """
        if (self.image.shape[0] != output_image_size[1]) or (
            self.image.shape[1] != output_image_size[0]
        ):
            self.image = cv2.resize(self.image, output_image_size)
