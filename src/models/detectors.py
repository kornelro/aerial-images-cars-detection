from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from src.data.image import Image
from src.data.sliders import Slider
from src.features.pipelines import Pipeline
from src.models.classifiers import Classifier, ProbClassifier

import logging
logger = logging.getLogger(__name__)


class Detector(ABC):

    def __init__(self):
        pass

    def __call__(
        self,
        image: Image,
        workers: int = 0,
        verbose: bool = True
    ) -> Tuple[Image, np.array, List[Set[float]]]:
        """
        Returns image with drawed detected boxes
        and list of detected boxes
        """

        bnd_boxes = self._detect(image, workers)
        image_array = image.image
        for bnd_box in bnd_boxes:
            image_array = cv2.rectangle(
                image_array,
                (bnd_box[4], bnd_box[5]),
                (bnd_box[2], bnd_box[3]),
                (0, 255, 0),
                2
            )

        return image_array, bnd_boxes, image

    @abstractmethod
    def _detect(
        self,
        image: np.array,
        workers: int = 0,
        verbose: bool = True
    ) -> List[Set[float]]:
        pass


class SliderDetector(Detector):

    def __init__(
        self,
        sliding_window: Slider,
        process_pipeline: Pipeline,
        classifier: Classifier,
    ):
        super().__init__()
        self.sliding_window = sliding_window
        self.process_pipeline = process_pipeline
        self.classifier = classifier

    def _detect(
        self,
        image: Image,
        workers: int = 0
    ) -> List[Set[float]]:

        detected_bnd_boxes = []

        if workers == 0:
            bnd_boxes = [b for b in self.sliding_window(image.image)]
            cropped_images = [
                self.process_pipeline.process(
                    image.get_car(b)
                ) for b in bnd_boxes
            ]
            prediction = self.classifier.predict(cropped_images)

            for i in range(len(prediction)):
                if prediction[i] == 1:
                    detected_bnd_boxes.append(bnd_boxes[i])

        else:
            bnd_boxes = [b for b in self.sliding_window(image.image)]

            with Pool(workers) as p:
                detected_bnd_boxes = p.map(
                    partial(
                        self._classify_image_for_parallel,
                        image=image
                    ),
                    bnd_boxes,
                )

            detected_bnd_boxes = [
                b for b in detected_bnd_boxes if b is not None
            ]

        return detected_bnd_boxes

    def _classify_image_for_parallel(
        self,
        bnd_box: Set[float],
        image: Image
    ):
        result = None

        cropped_image = image.get_car(bnd_box)
        features = self.process_pipeline.process(cropped_image)
        prediction = self.classifier.predict(features.reshape(1, -1))

        if prediction == 1:
            result = bnd_box

        return result


class SliderProbDetector(Detector):

    def __init__(
        self,
        sliding_window: Slider,
        process_pipeline: Pipeline,
        classifier: ProbClassifier,
        treshold: float = 0.5,
        nms_overlap: float = 0.5
    ):
        super().__init__()
        self.sliding_window = sliding_window
        self.process_pipeline = process_pipeline
        self.classifier = classifier
        self.treshold = treshold
        self.nms_overlap = nms_overlap

    def _detect(
        self,
        image: Image,
        workers: int = 0
    ) -> List[Set[float]]:

        if workers == 0:

            detected_bnd_boxes = []
            detected_bnd_boxes_probes = []

            bnd_boxes = [b for b in self.sliding_window(image.image)]
            cropped_images = [
                self.process_pipeline.process(
                    image.get_car(b)
                ) for b in bnd_boxes
            ]
            probes = self.classifier.predict(cropped_images)

            for i in range(len(probes)):
                if probes[i] > self.treshold:
                    detected_bnd_boxes.append(bnd_boxes[i])
                    detected_bnd_boxes_probes.append(probes[i])

        else:
            bnd_boxes = [b for b in self.sliding_window(image.image)]

            with Pool(workers) as p:
                results = p.map(
                    partial(
                        self._classify_image_for_parallel,
                        image=image
                    ),
                    bnd_boxes,
                )

            detected_bnd_boxes = [
                res[0] for res in results if res is not None
            ]
            detected_bnd_boxes_probes = [
                res[1] for res in results if res is not None
            ]

        return self._nms(
                detected_bnd_boxes,
                detected_bnd_boxes_probes,
                self.nms_overlap
            )

    def _classify_image_for_parallel(
        self,
        bnd_box: Set[float],
        image: Image
    ):
        result = None

        cropped_image = image.get_car(bnd_box)
        features = self.process_pipeline.process(cropped_image)
        prediction = self.classifier.predict(features.reshape(1, -1))[0]

        if prediction > self.treshold:
            result = (bnd_box, prediction)

        return result

    def _nms(
        self,
        detected_bnd_boxes: List[Set[float]],
        detected_bnd_boxes_probes: List[float],
        overlap: float
    ):

        print(len(detected_bnd_boxes))
        boxes = []
        for i in range(len(detected_bnd_boxes)):
            boxes.append(
                [
                    detected_bnd_boxes[i][4],  # x1
                    detected_bnd_boxes[i][5],  # y1
                    detected_bnd_boxes[i][2],  # x2
                    detected_bnd_boxes[i][3],  # y2
                    detected_bnd_boxes_probes[i]  # prob
                ]
            )

        # boxes is a list of size (n x 5)
        # trial is a numpy array of size (n x 5)
        # Author: Vicky

        if not boxes:
            pick = []
        else:
            trial = np.zeros((len(boxes), 5), dtype=np.float64)
            trial[:] = boxes[:]
            x1 = trial[:, 0]
            y1 = trial[:, 1]
            x2 = trial[:, 2]
            y2 = trial[:, 3]
            score = trial[:, 4]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            # vals = sort(score)
            i_array = np.argsort(score)
            pick = []
            count = 1
            while (i_array.size != 0):
                # print "Iteration:",count
                last = i_array.size
                i = i_array[last-1]
                pick.append(i)
                suppress = [last-1]
                for pos in range(last-1):
                    j = i_array[pos]
                    xx1 = max(x1[i], x1[j])
                    yy1 = max(y1[i], y1[j])
                    xx2 = min(x2[i], x2[j])
                    yy2 = min(y2[i], y2[j])
                    w = xx2 - xx1 + 1
                    h = yy2 - yy1 + 1
                    if (w > 0 and h > 0):
                        o = w * h / area[j]
                        # print("Overlap is", o)
                        if (o > overlap):
                            suppress.append(pos)
                i_array = np.delete(i_array, suppress)
                count = count + 1

        nms_bnd_boxes = []
        nms_bnd_boxes = list(map(lambda idx: detected_bnd_boxes[idx], pick))
        # print(len(nms_bnd_boxes))

        return nms_bnd_boxes
