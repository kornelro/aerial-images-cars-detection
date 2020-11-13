import cv2
from .image import Image
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Set


class ImageLoader(ABC):

    def __init__(self):
        pass

    def load_image(
        self,
        image_dir: str,
        annotation_dir: str
    ) -> None:

        image: np.array = cv2.imread(image_dir)
        with open(annotation_dir, 'r') as f:
            annotation_file: str = f.read()
            bnd_boxes = self.read_bnd_boxes(annotation_file, image)

        return Image(image, bnd_boxes)

    @abstractmethod
    def read_bnd_boxes(
        self,
        annotation_file: str,
        image: np.array
    ) -> List[Set[float]]:
        # returns list of sets with bnd boxes coords
        # (top_left_x, top_left_y, top_right_x, top_right_y
        # bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y)
        pass


class AerialCarsImageLoader(ImageLoader):

    def read_bnd_boxes(
        self,
        annotation_file: str,
        image: np.array
    ) -> List[Set[float]]:

        annotations = []

        for ann_row in annotation_file.split('\n'):
            ann_row = ann_row.split(' ')

            if ann_row[0] == '0':
                # TODO consider other classes

                xc = int(image.shape[1] * float(ann_row[1]))
                yc = int(image.shape[0] * float(ann_row[2]))
                w = int(image.shape[0] * float(ann_row[3]))
                h = int(image.shape[1] * float(ann_row[4]))

                top_left_x = xc - int(w/2)
                top_left_y = yc + int(h/2)

                top_right_x = xc + int(w/2)
                top_right_y = yc + int(h/2)

                bottom_left_x = xc - int(w/2)
                bottom_left_y = yc - int(h/2)

                bottom_right_x = xc + int(w/2)
                bottom_right_y = yc - int(h/2)

                annotations.append((
                    top_left_x, top_left_y,
                    top_right_x, top_right_y,
                    bottom_left_x, bottom_left_y,
                    bottom_right_x, bottom_right_y))

        return annotations
