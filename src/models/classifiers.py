from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np


class Classifier(ABC):

    @abstractmethod
    def predict(
        self,
        features: List[np.array]
    ) -> List[int]:
        pass


class ProbClassifier(ABC):

    @abstractmethod
    def predict(
        self,
        features: List[np.array]
    ) -> List[float]:
        """
        Return list of probabilities that bnd box is positive classified.
        """
        pass


class ProbSVMModelWrapper(ProbClassifier):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        probes = self.model.predict_proba(x)
        probes = list(map(lambda x: x[1], probes))

        return probes


class ResnetModelWrapper(ProbClassifier):

    def __init__(self, model):
        self.model = model

    def predict(self, images):
        # results = []
        probes_1 = []
        batches = np.array_split(images, round(len(images)/200))
        for batch in batches:
            batch = np.stack(batch, 0)
            print(batch.shape)
            preds = self.model.predict(batch)
            for pred in preds:
                # results.append(np.argmax(pred))
                probes_1.append(pred[1])
        return probes_1
