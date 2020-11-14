from multiprocessing import Pool
from typing import List, Tuple
from functools import partial

import numpy as np
from src.features.pipelines import RawImageToFeatures
from tqdm import tqdm


def build_features(
    images_with_classes: List[Tuple[np.array, int]],
    pipeline: RawImageToFeatures,
    parallel: bool = False,
    workers: int = 5
) -> List[Tuple[np.array, np.array, int]]:
    """
    Returns list of tuples contains original input images,
    feature vectors for input images and classes.
    Classes are stored in input and output tuples
    to enable parallel processing for multiclass data
    (doing parallel images order is shuffle).
    """

    features_with_classes = []
    if parallel:
        # TODO how to do parallel if we care about images order
        with Pool(workers) as p:
            features_with_classes = p.map(
                partial(
                    features_for_single_image,
                    pipeline=pipeline
                ),
                images_with_classes,

            )
    else:
        for image_with_class in tqdm(images_with_classes):
            features_with_classes.append((
                features_for_single_image(
                    image_with_class,
                    pipeline
                )
            ))

    return features_with_classes


def features_for_single_image(
    image_with_class: Tuple[np.array, int],
    pipeline: RawImageToFeatures
) -> Tuple[np.array, np.array, int]:
    return (
        image_with_class[0],
        pipeline.process(image_with_class[0]),
        image_with_class[1]
    )
