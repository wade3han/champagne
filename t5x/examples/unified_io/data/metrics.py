import json
import json
import logging
import os
import pickle
import re
import string
from collections import Counter, defaultdict
from os.path import join
from pathlib import Path
from typing import Sequence, List, Optional, Mapping, Any, Dict, Union

import gin
import numpy as np
import tensorflow as tf
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from scipy.stats import spearmanr
from seqio import Logger as SeqioLogger
from seqio.metrics import Scalar, Text

from t5x.examples.unified_io.data.data_utils import OTHER_INSTANCE_COLORS, BK_COLORS, \
  FIRST_OBJ_COLOR, get_default_vocabulary
from t5x.examples.unified_io.data.preprocessors import FINETUNE_IMAGE_INPUT_SIZE
from t5x.examples.unified_io.evaluator import UnifiedIOOutput
from t5x.examples.unified_io.metrics import rle
from t5x.examples.unified_io.metrics.bounding_box import BoundingBox, BBFormat, BBType
from t5x.examples.unified_io.metrics.coco_detection_evaluator import get_coco_summary
from t5x.examples.unified_io.metrics.depth import depth_example_metrics
from t5x.examples.unified_io.metrics.grit_keypoint import computeOks
from t5x.examples.unified_io.metrics.grit_localization import loc_metric, compute_iou
from t5x.examples.unified_io.metrics.grit_normal import sn_metric
from t5x.examples.unified_io.metrics.grit_vqa import preprocess_answer
from t5x.examples.unified_io.metrics.pascal_voc_evaluator import get_pascalvoc_metrics
from t5x.examples.unified_io.metrics.ptbtokenizer import PTBTokenizer
from t5x.examples.unified_io.metrics.utils import build_depth_prediction, extract_bboxes_from_text, \
  build_target_image, extract_points_from_text

from tensorflow.io import gfile

from seqio import metrics_lib

VOCAB = get_default_vocabulary()

def tokenize(captions: Dict[str, List[str]], ptb_tokenizer=False):
  """Tokenization captions, used for captioning metrics"""
  if ptb_tokenizer:
    # COCO's official tokenizer
    return PTBTokenizer().tokenize(captions)
  else:
    # NLTK's tokenizer, which is similar and does not require installing Java
    from nltk import TreebankWordTokenizer
    tok = TreebankWordTokenizer()
    return {
      k: [" ".join(tok.tokenize(cap)) for cap in v] for k, v in captions.items()
    }


def coco_captioning_metric(targets: Sequence[str], predictions: Sequence[Union[str, UnifiedIOOutput]], aux_values):
  if isinstance(targets[0], str):
    gts = {str(i): [p] for i, p in enumerate(targets)}
  elif isinstance(targets[0], dict):
    gts = {str(i): p["all_references"] for i, p in enumerate(targets)}
  elif isinstance(targets[0], list):
    gts = {}
    for i, p in enumerate(targets):
      if (not isinstance(p, list)) or (len(p) == 0):
        print(f"No reference for {i}...")
      else:
       gts[str(i)] = p
  else:
    raise NotImplementedError

  preds = {}
  for i, p in enumerate(predictions):
    if (not isinstance(targets[i], list)) or (len(targets[i]) == 0):
      continue
    if isinstance(predictions[0], str):
      preds[str(i)] = [p.lower()]
    else:
      preds[str(i)] = [p.text.lower()]

  gts = tokenize(gts)
  preds = tokenize(preds)

  scorers = {}
  scorers["cider"] = Cider()
  scorers["bleu"] = Bleu(4)
  scorers["meteor"] = Meteor()

  results = {}
  for name, scorer in scorers.items():
    if isinstance(scorer, Bleu):
      scores = scorer.compute_score(gts, preds, verbose=0)
      for i, score in enumerate(scores[0]):
        results[f"bleu{i+1}"] = score
    elif isinstance(scorer, Meteor):
      scores = scorer.compute_score(gts, preds)
      results["meteor"] = scores[0]
    else:
      score = scorer.compute_score(gts, preds)
      if isinstance(scorer, Cider):
        results["cider"] = score[0]
      else:
        raise NotImplementedError()

  scores = {k: Scalar(v) for k, v in results.items()}
  ixs = np.random.choice(len(targets), 20, replace=False)
  parts = [f"{predictions[i].text} (gt={targets[i][0]})" for i in ixs]
  print("\n".join(parts))
  scores["examples"] = Text(",  ".join(parts))
  return scores


def tagging_score(target, pred):
  if isinstance(target, list):
    target = Counter(target)
    return target[pred]
  else:
    return float(pred == target)


def image_tagging_metric(targets: Sequence[str], predictions: Sequence[UnifiedIOOutput]):
  score = np.mean([tagging_score(t, p.lower()) for t, p in zip(targets, predictions)])
  ixs = np.random.choice(len(targets), 20, replace=False)
  examples = [f"{predictions[i].lower()} (gt={targets[i][0]})" for i in ixs]
  return {
    "score": Scalar(score),
    "examples": Text(",  ".join(examples))
  }


def vqa_score(target, pred):
  pred = preprocess_answer(pred)
  if isinstance(target, list):
    target = Counter(preprocess_answer(x) for x in target)
    return min(target[pred] / 3.0, 1)
  else:
    return float(pred == target)


def vqa_metric(targets: Sequence[str], predictions: Sequence[UnifiedIOOutput]):
  score = np.mean([vqa_score(t, p.text.lower()) for t, p in zip(targets, predictions)])
  ixs = np.random.choice(len(targets), 20, replace=False)
  examples = [f"{predictions[i].text.lower()} (gt={targets[i][0]})" for i in ixs]
  return {
    "score": Scalar(score),
    "examples": Text(",  ".join(examples))
  }


def vcr_metric(targets: Sequence[str], predictions: Sequence[UnifiedIOOutput]):
  matches = [target == np.argmin(pred.scores) for target, pred in zip(targets, predictions)]
  return {
    "score": Scalar(np.mean(matches)),
  }


def exact_match(targets: Sequence, predictions: Sequence[str],
                aux_values, print_examples=True):
  if isinstance(targets[0], dict):
    targets = [x["text_target"] for x in targets]
  if isinstance(targets[0], np.ndarray):
    # Multiple correct answers
    matches = [pred.lower() in [x.decode("utf-8").lower() for x in target] for target, pred in zip(targets, predictions)]
  else:
    matches = [target.lower() == pred.lower() for target, pred in zip(targets, predictions)]
  if print_examples:
    ixs = np.random.choice(len(targets), min(20, len(targets)), replace=False)
    examples = [f"pred={predictions[i].lower()} gt={targets[i]}" for i in ixs]
    for ex in examples:
      print(ex)
  return {
    "score": Scalar(np.mean(matches)),
  }


def depth_metric(targets, predictions: List[UnifiedIOOutput], aux_values,
                 max_depth=10.0, scaled: Optional[bool]=False):
  stats = defaultdict(list)
  for example, pred in zip(targets, predictions):
    if scaled:
      raise NotImplementedError()
    else:
      # Undo the resizing and compare against the ground truth image
      gt_depth, image_info = example["depth"], example["image_info"]
      dec = build_depth_prediction(pred.image, image_info, max_depth)
      metrics = depth_example_metrics(gt_depth, dec, 0.001, max_depth)

    for k, v in metrics.items():
      stats[k].append(v)

  out = {}
  for k, v in stats.items():
    out[k] = Scalar(float(np.mean(v)))
  return out


def convert_image_to_instance_masks(img):
  if img.dtype == np.uint8:
    img = img.astype(np.int32)  # Avoid overflows
  img_flat = img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * 256 * 256
  bk = {0}  # Remove black background color
  instance_ids = set(img_flat.reshape(-1)).difference(bk)
  return [img_flat == x for x in instance_ids]


def clean_mask(mask, min_size):
  """Remove connected components that have less then `min_size` pixels"""
  from scipy import ndimage
  label, n_obj = ndimage.measurements.label(mask)
  cleaned = None
  for c in range(1, n_obj+1):
    is_c = label == c
    if np.sum(is_c) > min_size:
      if cleaned is None:
        cleaned = is_c
      else:
        cleaned = np.logical_or(cleaned, is_c)
  return cleaned


def get_segmentation_mask(img, segmention_mode) -> List[np.ndarray]:
  """Extract a list of binary segmentation masks from `img`"""
  if segmention_mode == "any_pixel":
    # Assume there is only a single instance
    is_instance = img.mean(-1) > 30
    return [is_instance]

  elif segmention_mode == "coarse_color":
    # Find instances based on coarse-grained color detection, and then clean them for
    # extra/floating background pixels. Pretty slow, I think because `clean_mask` is slow
    w, h = img.shape[:2]
    img = np.array(img).reshape((-1, 3))  # [n_pixels, 3]

    img = img.astype(np.float64)
    means = img.mean(axis=-1)
    mean_diff = img - means[:, None]

    # Background pixels are black or nearly black
    background = means <= 30

    # First object pixels are gray/white, we allow gray since the VAE will often put gray
    # pixels around the white blobs it is supposed to predict
    # We detect such pixels if all RGB values are close to the mean
    first_obj = np.logical_and(np.logical_not(background), np.abs(mean_diff).sum(-1) < 100)
    used = np.logical_and(background, first_obj)  # Pixel already assigned
    out = []
    first_obj = clean_mask(first_obj, 10)
    if np.any(first_obj):
      out.append(first_obj)

    color = np.argmax(img, -1)
    for c in range(3):
      # Find pixels if each color they must have that color's value
      # be the largest RGB value be large then the mean by a reasonable margin
      candidate = np.logical_and(np.logical_not(used), color == c)
      color_map = np.logical_and(candidate, np.abs(mean_diff[:, c]) > 40)
      color_map = clean_mask(color_map, 10)
      if np.any(color_map):
        out.append(color_map)
        used = np.logical_and(used, color_map)
    return [x.reshape(w, h) for x in out]

  elif segmention_mode == "color":
    h, w, _ = img.shape
    possible_colors = np.array(OTHER_INSTANCE_COLORS + [FIRST_OBJ_COLOR] + [BK_COLORS])  # [n_colors, 3]
    img = np.array(img).reshape((-1, 3))  # [n_pixels, 3]

    # [n_pixels, n_colors, 3]
    diffs = np.expand_dims(img, 1) - np.expand_dims(possible_colors, 0)
    diffs = np.abs(diffs).sum(-1)  # [n_pixels, n_colors]

    nearest = np.argmin(diffs, -1)  # [n_pixels]
    img = possible_colors[nearest]  # [n_pixels, 3]
    img =  img.reshape((h, w, 3))

    u_color = np.unique(img)
    masks = []
    for c in u_color:
      if c != 0:  # Background class
        masks.append(img == c)
    return masks

  elif segmention_mode == "any_disjoint":
    from scipy import ndimage
    is_instance = img.mean(-1) > 30
    values = is_instance.astype(np.int32)
    label, n_obj = ndimage.measurements.label(values)
    masks = []
    for c in range(1, n_obj+1):
      masks.append(label == c)
    return masks
  else:
    raise NotImplementedError()


def ref_exp_metric(targets, predictions: List[UnifiedIOOutput]):
  total_acc = 0
  total_iou = 0
  for target, pred in zip(targets, predictions):
    box, box_size = target  # Assumed to be xyxy scaled to the original image
    boxes, classes = extract_bboxes_from_text(pred.text, image_size=box_size)
    if len(boxes) == 0:
      pass  # No points
    else:
      # `box` is relative the input size, resize to be relative to the original size
      iou = compute_iou(boxes[0], box[0])
      total_iou += iou
      total_acc += float((iou > 0.5))

  n = len(predictions)
  return dict(acc=total_acc/n, iou=total_iou/n)


def undo_box_preprocessing(boxes, image_info):
  """"Convert boxes relative to the scaled/cropped inut image to pixel
  coordinates relative to the original image"""
  scaled_height = int(image_info[9])
  scaled_width = int(image_info[10])
  off_y = int(image_info[7])
  off_x = int(image_info[8])

  h, w = FINETUNE_IMAGE_INPUT_SIZE
  # Size of the valid/non-padding region of the image used as input
  crop_h = min(off_y + h, scaled_height) - off_y
  crop_w = min(off_x + w, scaled_width) - off_x

  # Clip boxes to the non-padded regions
  clip_ar = np.array([crop_h, crop_w, crop_h, crop_w])
  boxes = np.minimum(boxes, clip_ar.reshape(1, 4))

  # Undo the crop if there was one
  off_y, off_x = image_info[7], image_info[8]
  bias = np.array([off_y, off_x, off_y, off_x]).reshape(1, 4)
  boxes = boxes + bias

  # Undo the scaling
  inv_scale = image_info[2]
  return boxes * inv_scale


def grit_localization(targets, predictions: List[UnifiedIOOutput], original_scale=True):
  
  scores = []
  for target, pred in zip(targets, predictions):
    gt_boxes, image_info, src_boxes = target
    p_boxes, classes = extract_bboxes_from_text(pred.text, image_size=FINETUNE_IMAGE_INPUT_SIZE)

    if original_scale:
      # This should only affect the numbers in obscure cases where a boudnding
      # interacts with the crop region
      p_boxes = undo_box_preprocessing(p_boxes, image_info)
      h, w = image_info[3:5]
      gt_boxes = src_boxes * np.array([h, w, h, w]).reshape(1, 4)

    score = loc_metric(p_boxes, gt_boxes)
    scores.append(score)
  return {
    "score": Scalar(np.mean(scores))
  }


def surface_normal_metric(targets, predictions: List[UnifiedIOOutput],
                          rotate=True, ransac=True):
  """Computes GRIT sn metrics, not official"""
  total_score = 0
  html_table = []
  for target, pred in zip(targets, predictions):
    image_info = target["image_info"]
    pred_image = build_target_image(pred.image, image_info, to_int=True)
    gt_image = target["gt_image"]
    # TODO this a kind of a hack to find valid regions
    valid = np.logical_not(np.all(gt_image == 128, -1))
    score = sn_metric(pred_image, gt_image, valid, rotate=rotate, ransac=ransac)
    total_score += score

    row = {}
    row["gt_image"] = gt_image
    row["pred"] = pred_image
    row["score"] = f"{score:0.3f}"
    html_table.append(row)

  from t5x.examples.unified_io.build_html_visualizations import build_html_table
  html = build_html_table(html_table)
  with gfile.GFile("dbg.html", "w") as f:
    f.write(html)

  return {
    "score": Scalar(total_score/len(targets))
  }


def single_instance_keypoints(targets, predictions: List[UnifiedIOOutput]):
  """Scores for our person-region specific keypoint task using GRIT keypoint metric"""
  total_invalid = 0
  iou_total = 0
  for target, pred in zip(targets, predictions):
    points, invalid = extract_keypoints(pred.text)
    if invalid:
      total_invalid += 1
    if points is not None:
      gt = np.concatenate([target["keypoint_pos"], target["keypoint_label"]], 1)
      iou = computeOks(points.reshape((1, -1)), gt.reshape((1, -1)))
      iou_total += iou[0, 0]

  return {
    "score": Scalar(iou_total/len(targets)),
    "invalid": Scalar(total_invalid/len(targets))
  }


def coco_detection_metrics(targets, predictions):
  return detection_metrics(targets, predictions, "coco")


def pascal_detection_metrics(targets, predictions):
  return detection_metrics(targets, predictions, "pascal")


@gin.configurable()
class SaveMetrics(SeqioLogger):

  def __call__(self, task_name: str, step: Optional[int],
               metrics: Mapping[str, Any],
               dataset: Optional[tf.data.Dataset],
               inferences: Optional[Mapping[str, Sequence[Any]]],
               targets: Optional[Sequence[Any]]) -> None:
    if step is None:
      step = -1
    out = {}
    for metric_name, metric_value in metrics.items():
      if isinstance(metric_value, metrics_lib.Scalar):
        out[metric_name] = metric_value.value
      elif isinstance(metric_value, metrics_lib.Text):
        out[metric_name] = metric_value.textdata

    inferences_fname = os.path.join(self.output_dir, f"{task_name}-{step:06}-metrics.json")
    logging.info(f"Saving metrics to {inferences_fname}")
    with tf.io.gfile.GFile(inferences_fname, "w") as f:
      json.dump(out, f)


@gin.configurable()
class SaveTextPredictionsLogger(SeqioLogger):
  """Saves text output

  Assumes the post-processed output is a dictionary with a unique `example_id` field
  """

  def __call__(self, task_name: str, step: Optional[int],
               metrics: Mapping[str, Any],
               dataset: Optional[tf.data.Dataset],
               inferences: Optional[Mapping[str, Sequence[Any]]],
               targets: Optional[Sequence[Any]]) -> None:
    if step is None:
      step = -1
    out = {}
    for example_id, (text, input, target) in enumerate(zip(inferences["prediction"],
                                                           inferences["aux_value"]["text_inputs"],
                                                           targets)):
      data = dict(text=text, input=input, target=target)
      if "youtube_ids" in inferences["aux_value"]:
        data["youtube_id"] = inferences["aux_value"]["youtube_ids"][example_id].decode('utf-8')
        data["start_sec"] = int(inferences["aux_value"]["start_secs"][example_id])
        data["end_sec"] = int(inferences["aux_value"]["end_secs"][example_id])
      out[example_id] = data

    inferences_fname = os.path.join(self.output_dir, f"{task_name}-text-predictions.json")
    logging.info(f"Saving text predictions to {inferences_fname}")
    with tf.io.gfile.GFile(inferences_fname, "w") as f:
      json.dump(out, f)


def save_outputs(targets, predictions, output_dir):
  """"
  Metric that saves the targets/predictions to pickle, this takes up a lot of
  space so this is really just used for debugging or finding qualitive examples.
  """
  logging.info(f"Saving {len(targets)} examples to {output_dir}")
  prefix = join(output_dir, "example")
  voc = get_default_vocabulary()

  for i, (example, infer) in enumerate(zip(targets, predictions)):
    example_inputs = {}
    for k in [
      "image_targets",
      "image_target_masks",
      "image_inputs",
      "image_input_masks",
    ]:
      if k in example:
        example_inputs[k] = example[k]

    example_inputs["text_inputs"] = voc.decode(example["text_inputs"].tolist())
    example_inputs["text_target"] = voc.decode(example["text_targets"].tolist())

    example_outputs = {}
    if infer.image is not None:
      # the image will be bfloat16 cannot be pickled, convert to float32
      example_outputs["image"] = infer.image.astype(np.float32)

    example_outputs["text"] = infer.text
    example_outputs["image_tokens"] = infer.image_tokens
    example_outputs["scores"] = infer.scores
    out = dict(inputs=example_inputs, outputs=example_outputs)
    with gfile.GFile(prefix + f"{i}.pkl", "wb") as f:
      pickle.dump(out, f)
  logging.info(f"Examples were saved to {output_dir}")
  return {}


def extract_keypoints(text, image_info):
  points, labels = extract_points_from_text(text, image_size=FINETUNE_IMAGE_INPUT_SIZE)
  points = np.array(points)
  invalid = False  # IS this text a valid keypoint prediction

  # Convert label to integers
  for i, l in enumerate(labels):
    try:
      l = int(l) - 1
      if not (0 <= l <= 2):
        invalid = True
        l = 0
    except ValueError:
      invalid = True
      l = 0
    labels[i] = l
  labels = np.array(labels)
  if np.sum(labels) == 0:
    # No visible points predicted
    return None, invalid

  points = undo_box_preprocessing(np.tile(points, [1, 2]), image_info)[:, :2]
  points = points[:, ::-1]  # convert to xy

  # replace non visible point with mean so we do something non-crazy if the
  # GT turns out to be `visible`
  mean = np.mean(points[labels != 0], 0, keepdims=True)
  points[labels == 0] = mean

  if len(points) > 17:
    # Truncate if we generated extra for some reason
    invalid = True
    points = points[:17]
    labels = labels[:17]
  elif len(points) < 17:
    # Replace with mean if we generated too few
    invalid = True
    mean = np.mean(points, 0, keepdims=True)
    n = 17 - len(points)
    points = np.concatenate([points, np.tile(mean, (n, 1))], 0)
    # Doesn't matter for metrics anyway
    labels = np.concatenate([labels, np.zeros((n,), labels.dtype)])

  assert points.shape == (17, 2)
  points = np.concatenate([points, labels.astype(points.dtype)], -1)
  return points, invalid


@gin.configurable()
def save_predictions_vqa(
    targets, predictions, task="vqa", segmention_mode="coarse_color", output_dir='./grit_outputs'):
  """Save GRIT predictions in a format that can be directly submitted to the eval server"""

  logging.info("Staring compute grit predictions")
  data = []
  for ix, (target, pred) in enumerate(zip(targets, predictions)):
    example_id = target[0].decode("utf-8")
    image_info = target[1]

    text = pred
    # all_scores = float(np.exp(pred.scores))

    data.append({
      "example_id": example_id,
      "words": text,
      # "confidence": all_scores,
    })

  output_file = join(output_dir, f"{task}.json")
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  logging.info(f"Saving grit predictions to {output_file}")
  with gfile.GFile(output_file, "w") as f:
    json.dump(data, f, indent=2)
  return {}


@gin.configurable()
def save_predictions_dialogue(
    targets, predictions, output_dir='./grit_outputs'):
  """Save GRIT predictions in a format that can be directly submitted to the eval server"""

  logging.info("Staring compute grit predictions")
  data = []
  for ix, (target, pred) in enumerate(zip(targets, predictions)):
    text_targets = target[0]
    text_inputs = target[1]
    text = pred

    data.append({
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'prediction': text,
    })

  output_file = join(output_dir, "dialogue.json")
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  logging.info(f"Saving dialogue predictions to {output_file}")
  with gfile.GFile(output_file, "w") as f:
    json.dump(data, f, indent=2)
  return {}



@gin.configurable()
def save_predictions(
    targets, predictions, aux_values, task="vqa", segmention_mode="coarse_color", output_dir='./grit_outputs'):
  """Save GRIT predictions in a format that can be directly submitted to the eval server"""
  if not os.path.exists(output_dir) and "gs://" not in output_dir:
    os.mkdir(output_dir)
  logging.info("Staring compute grit predictions")
  data = []
  for ix, (target, pred, aux_value) in enumerate(zip(targets, predictions, aux_values)):
    text = pred.text

    data.append({
      "input": aux_value,
      "target": target,
      "prediction": text,
      # "confidence": all_scores,
    })

  output_file = join(output_dir, "prediction.json")
  logging.info(f"Saving predictions to {output_file}")
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  with gfile.GFile(output_file, "w") as f:
    json.dump(data, f, indent=2)
  return {}


# get log-probs for each class token.
def get_detection_prob(text, boxes, logprobs):
  tokens = VOCAB.encode(text)
  on = 0
  label_ixs = []
  while True:
    ixs = []
    if on == len(tokens):
      break
    while on < len(tokens) and 32000 <= tokens[on] <= 33000:
      on += 1
    if on == len(tokens):
      break
    while on < len(tokens) and tokens[on] < 32000:
      ixs.append(on)
      on += 1
    if ixs:
      label_ixs.append(ixs)

  if len(label_ixs) != len(boxes):
    assert len(label_ixs) == len(boxes) + 1
    label_ixs = label_ixs[:-1]

  all_scores = []
  for box, ixs in zip(boxes, label_ixs):
    score = [np.exp(logprobs[i]) for i in ixs]
    all_scores.append(np.mean(score))

  return all_scores


def detection_metrics(targets, predictions: List[UnifiedIOOutput], metrics="coco",
                      original_scale=True):
  # note we treat the boxes as xyxy even though they are yxyxy since yxyx are
  # not supported natively by `BoundingBox` and it doesnt change the scores
  for i, (target, pred) in enumerate(zip(targets, predictions)):
    gts = []
    boxes, labels, image_info, src_boxes = target
    if original_scale:
      h, w = image_info[3:5]
      boxes = src_boxes * np.array([h, w, h, w]).reshape(1, 4)

    image_name = f"image-{i}"
    for box, label in zip(boxes, labels):
      bb = BoundingBox(image_name, label.decode("utf-8"), coordinates=box, format=BBFormat.XYX2Y2)
      gts.append(bb)

    pred_boxes = []
    boxes, classes = extract_bboxes_from_text(pred.text, image_size=FINETUNE_IMAGE_INPUT_SIZE)
    if original_scale:
      boxes = undo_box_preprocessing(boxes, image_info)

    for box, cls in zip(boxes, classes):
      bb = BoundingBox(image_name, cls, coordinates=box, format=BBFormat.XYX2Y2,
                       bb_type=BBType.DETECTED, confidence=1.0)
      pred_boxes.append(bb)  # TODO is there way to get per-box confidence scores?

    if metrics == "coco":
      results = get_coco_summary(gts, pred_boxes)
      return {k: Scalar(v) for k, v in results.items()}
    elif metrics == "pascal":
      results = get_pascalvoc_metrics(gts, pred_boxes)
      return {"mAP": Scalar(results["mAP"])}
    else:
      raise NotImplementedError(metrics)
