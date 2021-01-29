import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from tqdm import tqdm

from src.data.image import Image
from src.data.image_loaders import ImageLoader


class DatasetLoader(ABC):

    def __init__(
        self,
        image_loader: ImageLoader
    ):
        self.image_loader = image_loader

    @abstractmethod
    def load_dataset(
        self,
        input_folder_filepath: str,
        images_files_types: List[str],
        annotations_files_types: List[str],
        verbose: bool
    ) -> List[Image]:
        pass


class AerialCarsDatasetLoader(DatasetLoader):

    def load_dataset(
        self,
        input_folder_filepath: str,
        images_files_types: Tuple[str],
        annotations_files_types: Tuple[str],
        verbose: bool = True
    ) -> List[Image]:

        tqdm_disable = not verbose
        if not input_folder_filepath.endswith('/'):
            input_folder_filepath = input_folder_filepath + '/'

        imgs = []

        for img_name in tqdm(
            os.listdir(input_folder_filepath),
            disable=tqdm_disable
        ):
            if img_name.endswith(images_files_types):
                img_dir = input_folder_filepath + img_name
                img_name = ''.join(
                    [img_name.split('.')[0], '.', annotations_files_types[0]]
                )
                ann_dir = input_folder_filepath + img_name
                imgs.append(self.image_loader.load_image(img_dir, ann_dir))

        return imgs


class VehiculesDatasetLoader(DatasetLoader):

    def load_dataset(
        self,
        input_folder_filepath: str,
        images_files_types: Tuple[str],
        annotations_files_types: Tuple[str],
        verbose: bool = True
    ) -> List[Image]:

        tqdm_disable = not verbose
        if not input_folder_filepath.endswith('/'):
            input_folder_filepath = input_folder_filepath + '/'

        imgs = []

        for img_name in tqdm(
            os.listdir(input_folder_filepath),
            disable=tqdm_disable
        ):
            if ('_co' in img_name) and (img_name.endswith(images_files_types)):
                img_dir = input_folder_filepath + img_name
                img_name = img_name.replace('_co', '')
                img_name = ''.join(
                    [img_name.split('.')[0], '.', annotations_files_types[0]]
                )
                ann_dir = input_folder_filepath + img_name
                try:
                    img_wit_ann = self.image_loader.load_image(
                        img_dir, ann_dir
                    )
                    if len(img_wit_ann.bnd_boxes) > 0:
                        imgs.append(img_wit_ann)
                except FileNotFoundError:
                    print('No file', ann_dir)
                except Exception:
                    print('Cannot read file', ann_dir)

        return imgs


class DOTADatasetLoader(DatasetLoader):

    def load_dataset(
        self,
        input_folder_filepath: str,
        images_files_types: Tuple[str],
        annotations_files_types: Tuple[str],
        verbose: bool = True
    ) -> List[Image]:

        tqdm_disable = not verbose
        if not input_folder_filepath.endswith('/'):
            input_folder_filepath = input_folder_filepath + '/'

        imgs = []

        for img_name in tqdm(
            os.listdir(input_folder_filepath),
            disable=tqdm_disable
        ):
            if img_name.endswith(images_files_types):
                img_dir = input_folder_filepath + img_name
                img_name = ''.join(
                    [img_name.split('.')[0], '.', annotations_files_types[0]]
                )
                ann_dir = input_folder_filepath + img_name
                try:
                    img_wit_ann = self.image_loader.load_image(
                        img_dir, ann_dir
                    )
                    if len(img_wit_ann.bnd_boxes) > 0:
                        imgs.append(img_wit_ann)
                except FileNotFoundError:
                    print('No file', ann_dir)

        return imgs


class OrtoDatasetLoader(DatasetLoader):

    def load_dataset(
        self,
        input_folder_filepath: str,
        images_files_types: Tuple[str],
        annotations_files_types: Tuple[str],
        verbose: bool = True
    ) -> List[Image]:

        tqdm_disable = not verbose
        if not input_folder_filepath.endswith('/'):
            input_folder_filepath = input_folder_filepath + '/'

        imgs = []

        for img_name in tqdm(
            os.listdir(input_folder_filepath),
            disable=tqdm_disable
        ):
            if img_name.endswith(images_files_types):
                img_dir = input_folder_filepath + img_name
                img_name = ''.join(
                    [img_name.split('.')[0], '.', annotations_files_types[0]]
                )
                ann_dir = input_folder_filepath + img_name
                try:
                    img_wit_ann = self.image_loader.load_image(
                        img_dir, ann_dir
                    )
                    if len(img_wit_ann.bnd_boxes) > 0:
                        imgs.append(img_wit_ann)
                except FileNotFoundError:
                    print('No file', ann_dir)
                except Exception:
                    print('Cannot read file', ann_dir)

        return imgs
