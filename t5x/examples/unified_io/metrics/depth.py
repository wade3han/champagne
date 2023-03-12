import numpy as np


def depth_example_metrics(gt_depth: np.ndarray, pred: np.ndarray, min_depth: float,
                          max_depth: float):
  stats = {}
  valid_mask = (np.logical_and(gt_depth > min_depth, gt_depth < max_depth)).reshape(-1)
  pred_depth = pred.reshape(-1)[valid_mask]
  gt_depth = gt_depth.reshape(-1)[valid_mask]
  gt_depth = np.clip(gt_depth, min_depth, max_depth)
  thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
  err = np.abs(np.log10(pred_depth) - np.log10(gt_depth))
  stats["rmse"] = np.sqrt(np.square(gt_depth - pred_depth).mean())
  stats["abs-rel"] = (np.abs(gt_depth - pred_depth) / gt_depth).mean()
  stats["log10"] = np.mean(err)
  stats["d1"] = (thresh < 1.25).mean()
  stats["d2"] = (thresh < 1.25 ** 2).mean()
  stats["d3"] = (thresh < 1.25 ** 3).mean()
  stats["sq-rel"] = (np.sqrt(np.square(gt_depth - pred_depth) / gt_depth).mean())
  return stats
