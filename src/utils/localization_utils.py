from functools import partial
import numpy as np


def get_annotation(bbox_list):
    bbox_display_list = []
    for b in bbox_list:
        x1, y1, x2, y2 = b
        bbox_display_list.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(bbox_display_list).astype(int)


class LocalisationTools:
    def __init__(self, metrics_name_list, image_shape=(224, 224), **kwargs):
        self._metrics_name_list = metrics_name_list
        self._image_shape = image_shape
        if len(self._metrics_name_list) == 0:
            print("No metrics set for evaluation...")
        self._set_metrics_list(**kwargs)

    def compute_metrics(self, binary_mask, annotation):
        if annotation.shape != binary_mask.shape:
            annotation = self._box_list_sanity_checks(annotation)
            ground_truth_mask = self._set_mask_from_box_list(annotation, binary_mask.shape)
        else:
            ground_truth_mask = np.where(annotation > 0.5, 1, 0)
        results = {}
        for name, metrics in zip(self._metrics_name_list, self._metrics_list):
            results[name] = metrics(ground_truth_mask, binary_mask)
        return results

    @staticmethod
    def _set_mask_from_box_list(box_list, shape: tuple = (224, 224)):
        mask = np.zeros(shape)
        for box in box_list:
            mask[box[1]: box[3], box[0]: box[2]] = 1
        return mask

    @staticmethod
    def _box_list_sanity_checks(box_list):
        box_list_ordered = []
        if not isinstance(box_list, np.ndarray):
            box_list = np.array(box_list)
        if len(box_list.shape) == 1:
            box_list = box_list[np.newaxis, :]
        for box in box_list:
            box_ordered = box.copy()
            box_ordered[:2] = np.minimum(box[:2], box[2:])
            box_ordered[2:] = np.maximum(box[:2], box[2:])
            box_list_ordered.append(box_ordered)
        return np.array(box_list_ordered).astype(int)

    def _set_metrics_list(self, **kwargs):
        self._metrics_list = []
        for name in self._metrics_name_list:
            if name == "iou":
                self._metrics_list.append(iou_score)
            elif name == "dice":
                self._metrics_list.append(dice_score)
            elif name == "all_iou":
                self._metrics_list.append(
                    all_iou_score)
            elif name == "correponding_iou":
                self._metrics_list.append(
                    correponding_iou_score)
            elif name == "ncc":
                self._metrics_list.append(
                    partial(ncc_score))


def dice_score(y_true, y_pred, eps=1e-8):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2 * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps)


def iou_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)


def correponding_iou_score(y_true, y_pred):
    ratio = np.mean(y_true) * 100
    t = np.percentile(y_pred, 100 - ratio)
    y_pred_binary = np.where(y_pred >= t, 1, 0)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_binary.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)


def all_iou_score(y_true, y_pred):
    iou_list = []
    for th in range(101):
        t = np.percentile(y_pred, th)
        y_pred_binary = np.where(y_pred >= t, 1, 0)
        y_true_f = y_true.flatten()
        y_pred_f = y_pred_binary.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        iou_list.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
    return np.array(iou_list).tolist()


def ncc_score(y_true, y_pred):
    y_true = (y_true - np.mean(y_true)) / np.std(y_true)
    y_pred = (y_pred - np.mean(y_pred)) / (np.std(y_pred) * len(y_pred.flatten()))
    return np.mean(np.correlate(y_pred.flatten(), y_true.flatten()))

