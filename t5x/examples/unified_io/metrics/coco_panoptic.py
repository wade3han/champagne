from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


OFFSET = 256 * 256 * 256
VOID = 0


def rgb2id(color):
  if isinstance(color, np.ndarray) and len(color.shape) == 3:
    if color.dtype == np.uint8:
      color = color.astype(np.int32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
  return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class PQStatCat:
  def __init__(self):
    self.iou = 0.0
    self.tp = 0
    self.fp = 0
    self.fn = 0

  def __iadd__(self, pq_stat_cat):
    self.iou += pq_stat_cat.iou
    self.tp += pq_stat_cat.tp
    self.fp += pq_stat_cat.fp
    self.fn += pq_stat_cat.fn
    return self


class PQStat:
  def __init__(self):
    self.pq_per_cat = defaultdict(PQStatCat)

  def __getitem__(self, i):
    return self.pq_per_cat[i]

  def __iadd__(self, pq_stat):
    for label, pq_stat_cat in pq_stat.pq_per_cat.items():
      self.pq_per_cat[label] += pq_stat_cat
    return self

  def pq_average(self, categories, isthing):
    pq, sq, rq, n = 0, 0, 0, 0
    per_class_results = {}
    for label, label_info in categories.items():
      if isthing is not None:
        cat_isthing = label_info['isthing'] == 1
        if isthing != cat_isthing:
          continue
      iou = self.pq_per_cat[label].iou
      tp = self.pq_per_cat[label].tp
      fp = self.pq_per_cat[label].fp
      fn = self.pq_per_cat[label].fn
      if tp + fp + fn == 0:
        per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
        continue
      n += 1
      pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
      sq_class = iou / tp if tp != 0 else 0
      rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
      per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
      pq += pq_class
      sq += sq_class
      rq += rq_class

    return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@dataclass
class SegmentInfo:
  id: int    # RGB key
  category_id: int  # Category name
  is_crowd: int = None
  area: int = None


def compute_pq_stats(pan_gt, gt_segment_info: List[Dict], pan_pred,
                     pred_segment_info: List[Dict], categories, pq_stat: PQStat):
  """
  segment_info: is a list of dictionaries containing: id, category_id and (for gt) is_crowd
  """
  pan_gt = rgb2id(pan_gt)
  pan_pred = rgb2id(pan_pred)

  gt_segms = {el["id"]: el for el in gt_segment_info}
  pred_segms = {el["id"]: el for el in pred_segment_info}
  pred_labels_set = set(el.id for el in pred_segment_info)

  labels, labels_cnt = np.unique(pan_pred, return_counts=True)
  for label, label_cnt in zip(labels, labels_cnt):
    if label not in pred_segms:
      if label == VOID:
        continue
      raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
    pred_segms[label]["area"] = label_cnt
    pred_labels_set.remove(label)
    if pred_segms[label]['category_id'] not in categories:
      raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
  if len(pred_labels_set) != 0:
    raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

  pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
  gt_pred_map = {}
  labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
  for label, intersection in zip(labels, labels_cnt):
    gt_id = label // OFFSET
    pred_id = label % OFFSET
    gt_pred_map[(gt_id, pred_id)] = intersection

  # count all matched pairs
  gt_matched = set()
  pred_matched = set()
  for label_tuple, intersection in gt_pred_map.items():
    gt_label, pred_label = label_tuple
    if gt_label not in gt_segms:
      continue
    if pred_label not in pred_segms:
      continue
    if gt_segms[gt_label]['iscrowd'] == 1:
      continue
    if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
      continue

    union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
    iou = intersection / union
    if iou > 0.5:
      pq_stat[gt_segms[gt_label]['category_id']].tp += 1
      pq_stat[gt_segms[gt_label]['category_id']].iou += iou
      gt_matched.add(gt_label)
      pred_matched.add(pred_label)

  # count false positives
  crowd_labels_dict = {}
  for gt_label, gt_info in gt_segms.items():
    if gt_label in gt_matched:
      continue
    # crowd segments are ignored
    if gt_info['iscrowd'] == 1:
      crowd_labels_dict[gt_info['category_id']] = gt_label
      continue
    pq_stat[gt_info['category_id']].fn += 1

  # count false positives
  for pred_label, pred_info in pred_segms.items():
    if pred_label in pred_matched:
      continue
    # intersection of the segment with VOID
    intersection = gt_pred_map.get((VOID, pred_label), 0)
    # plus intersection with corresponding CROWD region if it exists
    if pred_info['category_id'] in crowd_labels_dict:
      intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
    # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
    if intersection / pred_info['area'] > 0.5:
      continue
    pq_stat[pred_info['category_id']].fp += 1


def compute_stats(pq_stat: PQStat, categories):
  metrics = [("All", None), ("Things", True), ("Stuff", False)]
  all_results = {}
  for name, isthing in metrics:
    av_results, per_class_results = pq_stat.pq_average(categories, isthing=isthing)
    for k, v in av_results.items():
      all_results[f"{name}-{k}"] = v
  return all_results

