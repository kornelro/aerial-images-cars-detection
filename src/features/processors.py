from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
from skimage.transform import resize


class Processor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def process(
        self,
        image: np.array
    ) -> np.array:
        """ Returns processed input image """
        pass


class Resize(Processor):

    def __init__(
        self,
        output_size: Tuple[int, int]
    ):
        super().__init__()
        self.output_size = output_size

    def process(
        self,
        image: np.array
    ) -> np.array:

        image = resize(image, self.output_size)
        image = np.float32(image)

        return image


class Rotate90(Processor):

    def __init__(self):
        super().__init__()

    def process(
        self,
        image: np.array
    ) -> np.array:

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image


class RotateToHorizontal(Processor):

    def __init__(self):
        super().__init__()
        self.rotate = Rotate90()

    def process(
        self,
        image: np.array
    ) -> np.array:

        if image.shape[0] > image.shape[1]:
            image = self.rotate.process(image)

        return image


class NormImage(Processor):

    def __init__(
        self,
        max_v: int = 256
    ):
        super().__init__()
        self.max_v = max_v

    def process(
        self,
        image: np.array
    ) -> np.array:

        return image / self.max_v
