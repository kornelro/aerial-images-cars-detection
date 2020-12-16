# -*- coding: utf-8 -*-
from src.features.build_features import build_features
from src.data.dataset_loaders import DatasetLoader
from typing import List, Tuple
import click
import logging
import random
from tqdm import tqdm
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from ..features.pipelines import RawImageToFeatures


# @click.command()
# @click.argument('input_folder_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# @click.argument('dataset_loader', type=DatasetLoader)
# @click.argument('images_files_type', type=Tuple[str])
# @click.argument('annotations_files_type', type=Tuple[str])
# @click.argument('process_pipeline', type=RawImageToFeatures)
# @click.argument('negative_images_size', type=Tuple[int, int])
# @click.argument('negative_examples_size', type=float)
# @click.argument('workers', type=int)
# @click.argument('verbose', type=bool)
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

    # TODO save to file
    return data


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     make_dataset()
