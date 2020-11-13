from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from src.features.pipelines import RawImageToFeatures
from tqdm import tqdm


def build_features(
    images_with_classes: List[Tuple[np.array, int]],
    pipeline: RawImageToFeatures,
    parallel: bool = False,
    workers: int = 5
) -> List[Tuple[np.array, int]]:
    """
    Returns list of tuples contains feature vectors for input
    images and classes. Classes are stored in input and output
    tuples to enable parallel processing for multiclass data
    (doing parallel images order is shuffle).
    """

    features_with_classes = []
    if parallel:
        # TODO how to do parallel if we care about images order
        with Pool(workers) as p:
            features_with_classes = p.map(
                build_features_for_single_input,
                images_with_classes
            )
    else:
        for image_with_class in tqdm(images_with_classes):
            features_with_classes.append(
                build_features_for_single_input(image_with_class)
            )

    return features_with_classes


def build_features_for_single_input(
    image_with_class: Tuple[np.array, int],
    pipeline: RawImageToFeatures
) -> Tuple[np.array, int]:
    return (pipeline.process(image_with_class[0]), image_with_class[1])
