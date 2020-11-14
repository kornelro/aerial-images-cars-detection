from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog


class Descriptor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def process(
        self,
        image: np.array
    ) -> Tuple[np.array, np.array]:
        """ Returns feature vector and processed image for given image """
        pass


class HOGDescriptor(ABC):

    def __init__(
        self,
        orientations: int = 9,
        cells_per_block: Tuple[int, int] = (2, 2),
        pixels_per_cell: Tuple[int, int] = (4, 4),
        multichannel: bool = True,
        visualize: bool = True
    ):
        super().__init__()
        self.orientations = orientations
        self.cells_per_block = cells_per_block
        self.pixels_per_cell = pixels_per_cell
        self.multichannel = multichannel
        self.visualize = visualize

    def process(
        self,
        image: np.array
    ) -> Tuple[np.array, np.array]:

        fd, hog_image = hog(
            image,
            orientations=self.orientations,
            cells_per_block=self.cells_per_block,
            pixels_per_cell=self.pixels_per_cell,
            multichannel=True,
            visualize=self.visualize
        )

        return (fd, hog_image)


class LBPDescriptor(ABC):

    def __init__(
        self,
        bins: int = 256,
        range: Tuple[int, int] = (0, 256)
    ):
        super().__init__()
        self.bins = bins
        self.range = range

    def process(
        self,
        image: np.array
    ) -> Tuple[np.array, np.array]:

        height, width, _ = image.shape

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width), np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = self._lbp_calculated_pixel(img_gray, i, j)

        histogram, bin_edges = np.histogram(
            img_lbp,
            bins=self.bins,
            range=self.range
        )

        return (histogram, img_lbp)

    def _get_pixel(self, img, center, x, y):

        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except IndexError:
            pass

        return new_value

    def _lbp_calculated_pixel(self, img, x, y):

        center = img[x][y]
        val_ar = []
        val_ar.append(self._get_pixel(img, center, x-1, y-1))
        val_ar.append(self._get_pixel(img, center, x-1, y))
        val_ar.append(self._get_pixel(img, center, x-1, y + 1))
        val_ar.append(self._get_pixel(img, center, x, y + 1))
        val_ar.append(self._get_pixel(img, center, x + 1, y + 1))
        val_ar.append(self._get_pixel(img, center, x + 1, y))
        val_ar.append(self._get_pixel(img, center, x + 1, y-1))
        val_ar.append(self._get_pixel(img, center, x, y-1))

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]

        return val
