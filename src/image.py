from typing import List, Set

import cv2
import numpy as np


class Image:

    def __init__(
        self,
        image: np.array,
        bnd_boxes: List[Set[float]]
    ) -> None:

        self.image = image
        self.bnd_boxes = bnd_boxes

    def get_car(
        self,
        bnd_box
    ) -> np.array:
        #  return cropped bounding box
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
