from typing import Tuple, List, Set, Any
import numpy as np
from src.data.dataset_loaders import DatasetLoader
from src.models.detectors import Detector
from tqdm import tqdm
from copy import deepcopy
import cv2


def is_intersecting(box1, box2) -> bool:
    box1_xtl = box1[4]
    box1_ytl = box1[5]
    box1_xbr = box1[2]
    box1_ybr = box1[3]

    box2_xtl = box2[4]
    box2_ytl = box2[5]
    box2_xbr = box2[2]
    box2_ybr = box2[3]

    if box1_xtl > box2_xbr:
        return False  # boxA is right of boxB
    if box2_xtl > box1_xbr:
        return False  # boxA is left of boxB
    if box1_ybr < box2_ytl:
        return False  # boxA is above boxB
    if box1_ytl > box2_ybr:
        return False  # boxA is below boxB
    return True


def get_intersection_area(box1, box2) -> float:
    box1_xtl = box1[4]
    box1_ytl = box1[5]
    box1_xbr = box1[2]
    box1_ybr = box1[3]

    box2_xtl = box2[4]
    box2_ytl = box2[5]
    box2_xbr = box2[2]
    box2_ybr = box2[3]

    xtl = max(box1_xtl, box2_xtl)
    ytl = max(box1_ytl, box2_ytl)
    xbr = min(box1_xbr, box2_xbr)
    ybr = min(box1_ybr, box2_ybr)
    # intersection area
    return (xbr - xtl + 1) * (ybr - ytl + 1)


def area(box) -> float:
    xtl = box[4]
    ytl = box[5]
    xbr = box[2]
    ybr = box[3]
    return (xbr - xtl + 1) * (ybr - ytl + 1)


def get_union_areas(box1, box2, intersection_area: float = None) -> float:
    if intersection_area is None:
        intersection_area = get_intersection_area(box1, box2)
    return float(area(box1) + area(box2) - intersection_area)


def intersection_over_union(box1, box2) -> float:
    """
    Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
    two bounding boxes.
    """
    # if boxes dont intersect
    if not is_intersecting(box1, box2):
        return 0
    intersection = get_intersection_area(box1, box2)
    union = get_union_areas(box1, box2, intersection_area=intersection)
    # intersection over union
    iou = intersection / union
    assert iou >= 0, '{} = {} / {}, {}, {}'.format(iou, intersection, union, box1, box2)
    return iou


def precision(tp: float, fp: float) -> float:
    return tp / (tp + fp)


def recall(tp: float, fn: float) -> float:
    return tp / (tp + fn)


def f1_score(tp: float, fp: float, fn: float) -> float:
    return tp / (tp + 0.5*(fp + fn))


def evaluate_image(
        image,
        pred_bnd_boxes: List[Set[float]],
        iou_threshold: float
) -> Tuple[Any, Any]:
    true_bnd_boxes = image.bnd_boxes

    tps = np.zeros(len(pred_bnd_boxes))
    fps = np.zeros(len(pred_bnd_boxes))

    from collections import Counter
    repeats_counter = Counter()

    for idp, pred_box in enumerate(pred_bnd_boxes):
        pred_max_iou = 0
        best_true_matched = -1
        for idt, true_box in enumerate(true_bnd_boxes):
            iou = intersection_over_union(pred_box, true_box)
            if iou > pred_max_iou:
                pred_max_iou = iou
                best_true_matched = idt
        if pred_max_iou >= iou_threshold:
            if repeats_counter[best_true_matched] == 0:
                tps[idp] = 1
                repeats_counter[best_true_matched] = 1
            else:
                fps[idp] = 1
        else:
            fps[idp] = 1

    tp = np.sum(tps)
    fp = np.sum(fps)

    return tp, fp


def validate_model(
        dataset_loader: DatasetLoader,
        input_folder_filepath: str,
        images_files_types: Tuple[str],
        annotations_files_types: Tuple[str],
        detector: Detector,
        output_folder_filepath: str,
        workers: int = 0,
        iou_threshold=0.5
):
    images = dataset_loader.load_dataset(
        input_folder_filepath,
        images_files_types,
        annotations_files_types,
        False
    )

    processed_images = []

    for image in tqdm(images):
        processed_images.append(
            detector(deepcopy(image), workers)
        )

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for id, processed_image in enumerate(processed_images):
        image_array, pred_bnd_boxes, image = processed_image
        tp, fp = evaluate_image(image, pred_bnd_boxes, iou_threshold)
        true_positives += tp
        false_positives += fp
        false_negatives += (len(image.bnd_boxes) - tp)

        image_filename = f'prediction_output_{id}.jpg'
        output_full_path = output_folder_filepath + '/' + image_filename
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_full_path, image_array)

    return true_positives, false_positives, false_negatives, processed_images