from typing import List, Set
from random import random

import cv2
import numpy as np


class Image:

    def __init__(
        self,
        image: np.array,
        bnd_boxes: List[Set[float]],
        image_dir: str
    ) -> None:

        self.image = image
        self.bnd_boxes = bnd_boxes
        self.image_dir = image_dir

    def get_car(
        self,
        bnd_box
    ) -> np.array:
        #  returns cropped bounding box
        #  allows to crop rotated bnd boxes

        img = self.image
        cnt = np.array([
                [[bnd_box[4], bnd_box[5]]],
                [[bnd_box[0], bnd_box[1]]],
                [[bnd_box[2], bnd_box[3]]],
                [[bnd_box[6], bnd_box[7]]]
            ])

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(img, M, (width, height))

        return warped

    def get_random_box(
        self,
        width: int,
        height: int
    ) -> np.array:
        #  returns image cropped by random bnd box

        xc = int(self.image.shape[1] * random())
        yc = int(self.image.shape[0] * random())
        w = width
        h = height

        top_left_x = xc - int(w/2)
        top_left_y = yc + int(h/2)

        top_right_x = xc + int(w/2)
        top_right_y = yc + int(h/2)

        bottom_left_x = xc - int(w/2)
        bottom_left_y = yc - int(h/2)

        bottom_right_x = xc + int(w/2)
        bottom_right_y = yc - int(h/2)

        return self.get_car((
            top_left_x, top_left_y,
            top_right_x, top_right_y,
            bottom_left_x, bottom_left_y,
            bottom_right_x, bottom_right_y))

    def get_negative_box(
        self,
        width: int,
        height: int
    ) -> np.array:
        """
        Returs negative image (negative bnd_box not overlaped with positive bnd box).
        """

        found = False

        while not found:

            xc = int(self.image.shape[1] * random())
            yc = int(self.image.shape[0] * random())
            w = width
            h = height

            top_left_x = xc - int(w/2)
            top_left_y = yc + int(h/2)

            top_right_x = xc + int(w/2)
            top_right_y = yc + int(h/2)

            bottom_left_x = xc - int(w/2)
            bottom_left_y = yc - int(h/2)

            bottom_right_x = xc + int(w/2)
            bottom_right_y = yc - int(h/2)

            candidate_bnd_box = (
                top_left_x, top_left_y,
                top_right_x, top_right_y,
                bottom_left_x, bottom_left_y,
                bottom_right_x, bottom_right_y
            )

            intersect = False
            for bnd_box in self.bnd_boxes:
                if self._intersect(
                    bnd_box, candidate_bnd_box
                ):
                    intersect = True

            if not intersect:
                found = True

        return self.get_car(candidate_bnd_box)

    def _intersect(
        self,
        bnd_box1,
        bnd_box2
    ):
        intersect = True

        if bnd_box1[5] > bnd_box2[1]:
            intersect = False
        elif bnd_box1[1] < bnd_box2[1]:
            intersect = False
        elif bnd_box1[2] < bnd_box2[0]:
            intersect = False
        elif bnd_box1[0] > bnd_box2[2]:
            intersect = False

        return intersect
