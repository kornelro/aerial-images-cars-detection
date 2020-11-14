from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from src.data.image import Image
from src.data.sliders import Slider
from src.features.pipelines import RawImageToFeatures
from src.models.classifiers import Classifier
from tqdm import tqdm

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
    ) -> Tuple[np.array, List[Set[float]]]:
        """
        Returns image with drawed detected boxes
        and list of detected boxes
        """

        bnd_boxes = self.detect(image, workers, verbose)
        image = image.image
        for bnd_box in bnd_boxes:
            image = cv2.rectangle(
                image,
                (bnd_box[4], bnd_box[5]),
                (bnd_box[2], bnd_box[3]),
                (0, 255, 0),
                2
            )

        return (image, bnd_boxes)

    @abstractmethod
    def detect(
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
        process_pipeline: RawImageToFeatures,
        classifier: Classifier
    ):
        super().__init__()
        self.sliding_window = sliding_window
        self.process_pipeline = process_pipeline
        self.classifier = classifier

    def detect(
        self,
        image: Image,
        workers: int = 0,
        verbose: bool = True
    ) -> List[Set[float]]:

        tqdm_dsiable = not verbose

        detected_bnd_boxes = []

        if workers == 0:
            for bnd_box in tqdm(
                self.sliding_window(image.image),
                disable=tqdm_dsiable
            ):

                cropped_image = image.get_car(bnd_box)
                features = self.process_pipeline.process(cropped_image)
                prediction = self.classifier.predict(features.reshape(1, -1))

                if prediction == 1:
                    detected_bnd_boxes.append(bnd_box)

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
