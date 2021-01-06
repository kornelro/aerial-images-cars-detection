# -*- coding: utf-8 -*-
import logging
import pickle
import random
from typing import Tuple

from tqdm import tqdm

from ..features.build_features import build_features
from ..features.pipelines import RawImageToFeatures
from .dataset_loaders import DatasetLoader


def make_dataset(
    input_folder_filepath: str,
    output_filepath: str,
    dataset_loader: DatasetLoader,
    images_files_types: Tuple[str],
    annotations_files_types: Tuple[str],
    process_pipeline: RawImageToFeatures,
    negative_images_size: Tuple[int, int],
    negative_examples_size: float = 0.5,
    workers: int = 0,
    verbose: bool = True
):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    tqdm_disable = not verbose

    logger.info('Reading images...')
    images = dataset_loader.load_dataset(
        input_folder_filepath,
        images_files_types,
        annotations_files_types,
        verbose
    )

    data = []
    logger.info('Cropping cars...')
    for image in tqdm(images, disable=tqdm_disable):
        for bnd_box in image.bnd_boxes:
            data.append((
                image.get_car(bnd_box),
                1
            ))

    logger.info('Cropping random boxes...')
    positive_examples_size = 1 - negative_examples_size
    negative_examples_number = int(positive_examples_size \
        * len(data) / negative_examples_size)
    for i in tqdm(
        range(negative_examples_number),
        disable=tqdm_disable
    ):
        image = random.choice(images)
        data.append((
            image.get_random_box(
                negative_images_size[0],
                negative_images_size[1]
            ),
            0
        ))

    logger.info('Processing images...')
    data = build_features(
        images_with_classes=data,
        pipeline=process_pipeline,
        parallel=workers > 0,
        workers=workers
    )

    logger.info('Saving pickle...')
    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)

    return data
