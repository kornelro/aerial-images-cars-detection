from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Classifier(ABC):

    @abstractmethod
    def predict(
        self,
        features: np.array
    ) -> int:
        pass


class NNClassifier(ABC):

    @abstractmethod
    def predict(
        self,
        features: List[np.array]
    ) -> List[int]:
        pass
