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


class Blur(Processor):

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5)
    ):
        super().__init__()
        self.kernel_size = kernel_size

    def process(
        self,
        image: np.array
    ) -> np.array:

        image = cv2.GaussianBlur(
            image,
            self.kernel_size,
            cv2.BORDER_DEFAULT)

        return image


class EqualHist(Processor):

    def __init__(
        self
    ):
        super().__init__()

    def process(
        self,
        image: np.array
    ) -> np.array:

        R, G, B = cv2.split(image)

        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        equ = cv2.merge((output1_R, output1_G, output1_B))

        return equ
