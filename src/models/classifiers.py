from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):

    @abstractmethod
    def predict(
        self,
        features: np.array
    ) -> int:
        pass
