import cv2
from .image import Image
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Set, Tuple


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


class AerialCarsSquareImageLoader(ImageLoader):

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

                if w > h:
                    k = w
                else:
                    k = h

                top_left_x = xc - int(k/2)
                top_left_y = yc + int(k/2)

                top_right_x = xc + int(k/2)
                top_right_y = yc + int(k/2)

                bottom_left_x = xc - int(k/2)
                bottom_left_y = yc - int(k/2)

                bottom_right_x = xc + int(k/2)
                bottom_right_y = yc - int(k/2)

                annotations.append((
                    top_left_x, top_left_y,
                    top_right_x, top_right_y,
                    bottom_left_x, bottom_left_y,
                    bottom_right_x, bottom_right_y))

        return annotations


class AerialCarsFixedSizeImageLoader(ImageLoader):

    def __init__(
        self,
        bnd_box_size: Tuple[int, int]
    ):
        super().__init__()
        self.bnd_box_size = bnd_box_size

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
                w = self.bnd_box_size[0]
                h = self.bnd_box_size[1]

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


class VehiculesImageLoader(ImageLoader):
    def read_bnd_boxes(
            self,
            annotation_file: str,
            image: np.array
    ) -> List[Set[float]]:

        annotations = []

        for ann_row in annotation_file.split('\n'):
            if len(ann_row) > 0:
                ann_row = ann_row.split(' ')

                # rotation = ann_row[2]
                # vehicle_class = ann_row[3]
                # is_whole = ann_row[4]
                # is_occluded = ann_row[5]
                #
                # x_center = float(ann_row[0])
                # y_center = float(ann_row[1])

                x_1, y_1 = int(ann_row[6]), int(ann_row[6 + 4])
                x_2, y_2 = int(ann_row[7]), int(ann_row[7 + 4])
                x_3, y_3 = int(ann_row[8]), int(ann_row[8 + 4])
                x_4, y_4 = int(ann_row[9]), int(ann_row[9 + 4])

                polygon = [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]

                result = [val for sublist in polygon for val in sublist]
                annotations.append(tuple(result))

                # points = np.array(polygon, np.int32)
                #
                # top_left_x, top_left_y, height, width = cv2.boundingRect(points)
                #
                # top_right_x = top_left_x + width
                # top_right_y = top_left_y
                #
                # bottom_left_x = top_left_x
                # bottom_left_y = top_left_y + height
                #
                # bottom_right_x = top_left_x + width
                # bottom_right_y = top_left_y + height
                #
                # annotations.append((
                #     top_left_x, top_left_y,
                #     top_right_x, top_right_y,
                #     bottom_left_x, bottom_left_y,
                #     bottom_right_x, bottom_right_y))

        return annotations


class VehiculesSquareImageLoader(ImageLoader):
    def read_bnd_boxes(
            self,
            annotation_file: str,
            image: np.array
    ) -> List[Set[float]]:

        annotations = []

        for ann_row in annotation_file.split('\n'):
            if (len(ann_row) > 0):
                ann_row = ann_row.split(' ')

                if (
                    (ann_row[3] == '1')  # object is a car
                    and (ann_row[4] == '1')  # entirely in image
                    and (ann_row[5] == '0')  # not occluded
                ):
                    # TODO consider other values
                    # TODO similar ifs for othe Vehicules loaders

                    xc = int(float(ann_row[0]))
                    yc = int(float(ann_row[1]))

                    x_1, y_1 = int(ann_row[6]), int(ann_row[6 + 4])
                    x_2, y_2 = int(ann_row[7]), int(ann_row[7 + 4])
                    x_3, y_3 = int(ann_row[8]), int(ann_row[8 + 4])
                    x_4, y_4 = int(ann_row[9]), int(ann_row[9 + 4])

                    polygon = [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]
                    points = np.array(polygon, np.int32)
                    top_left_x, top_left_y, h, w = cv2.boundingRect(points)

                    if w > h:
                        k = w
                    else:
                        k = h

                    top_left_x = xc - int(k/2)
                    top_left_y = yc + int(k/2)

                    top_right_x = xc + int(k/2)
                    top_right_y = yc + int(k/2)

                    bottom_left_x = xc - int(k/2)
                    bottom_left_y = yc - int(k/2)

                    bottom_right_x = xc + int(k/2)
                    bottom_right_y = yc - int(k/2)

                    annotations.append((
                        top_left_x, top_left_y,
                        top_right_x, top_right_y,
                        bottom_left_x, bottom_left_y,
                        bottom_right_x, bottom_right_y))

        return annotations


class VehiculesFixedSizeImageLoader(ImageLoader):

    def __init__(
            self,
            bnd_box_size: Tuple[int, int]
    ):
        super().__init__()
        self.bnd_box_size = bnd_box_size

    def read_bnd_boxes(
            self,
            annotation_file: str,
            image: np.array
    ) -> List[Set[float]]:

        annotations = []

        for ann_row in annotation_file.split('\n'):
            if len(ann_row) > 0:
                ann_row = ann_row.split(' ')

                # rotation = ann_row[2]
                # vehicle_class = ann_row[3]
                # is_whole = ann_row[4]
                # is_occluded = ann_row[5]

                xc = int(float(ann_row[0]))
                yc = int(float(ann_row[1]))

                w = self.bnd_box_size[0]
                h = self.bnd_box_size[1]

                top_left_x = xc - int(w / 2)
                top_left_y = yc + int(h / 2)

                top_right_x = xc + int(w / 2)
                top_right_y = yc + int(h / 2)

                bottom_left_x = xc - int(w / 2)
                bottom_left_y = yc - int(h / 2)

                bottom_right_x = xc + int(w / 2)
                bottom_right_y = yc - int(h / 2)

                annotations.append((
                    top_left_x, top_left_y,
                    top_right_x, top_right_y,
                    bottom_left_x, bottom_left_y,
                    bottom_right_x, bottom_right_y))

        return annotations
