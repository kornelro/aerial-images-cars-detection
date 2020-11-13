from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .processors import Processor

from.descriptors import Descriptor


class Pipeline(ABC):

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def process(
        self,
        image: np.array
    ) -> np.array:
        """ Returns features vector for raw image """
        pass


class RawImageToFeatures(Pipeline):

    def __init__(
        self,
        processors: List[Processor],
        descriptors: List[Descriptor]
    ):
        super().__init__()
        self.processors = processors
        self.descriptors = descriptors

    def process(
        self,
        image: np.array
    ) -> np.array:

        for processor in self.processors:
            image = processor.process(image)

        features = np.array([])
        for descriptor in self.descriptors:
            features = np.concatenate(
                features, descriptor.process(image)
            )

        return features
