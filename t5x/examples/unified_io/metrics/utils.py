import pdb
from typing import List, Tuple

import numpy as np
import re
import tensorflow as tf

from t5x.examples.unified_io.data.data_utils import load_class_name, denormalize_boxes, \
  OTHER_INSTANCE_COLORS, BK_COLORS, FIRST_OBJ_COLOR, get_default_vocabulary

VOCAB = get_default_vocabulary()


bbox_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(4)) + " ?([^<]+)")
point_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(2)) + " ?([^<]+)")


def extract_bboxes_from_text(text, image_size=None, num_bin=1000, vocab_start=100) -> Tuple[np.ndarray, List[str]]:
  """Extract boxes mentioned in `text` using our location encoding"""
  return extract_coordinates_from_text(text, image_size, num_bin, vocab_start, 4)


def extract_points_from_text(text, image_size=None, num_bin=1000, vocab_start=100) -> Tuple[np.ndarray, List[str]]:
  """Extract points mentioned in `text` using our location encoding"""
  return extract_coordinates_from_text(text, image_size, num_bin, vocab_start, 2)


def extract_coordinates_from_text(text, image_size=None, num_bin=1000,
                                  vocab_start=100, n_coordinates=2):
  boxes = []
  class_names = []
  if n_coordinates == 2:
    exp = point_regex
  elif n_coordinates == 4:
    exp = bbox_regex
  else:
    raise NotImplementedError()
  
  for match in exp.finditer(text):
    token_ids = [int(match.group(i)) for i in range(1, 1+n_coordinates)]
    if not all(vocab_start <= ix < (vocab_start+num_bin) for ix in token_ids):
      # Contains non-location token ids
      continue
    class_names.append(match.group(n_coordinates+1).strip())
    boxes.append(token_ids)

  if not boxes:
    return np.zeros((0, n_coordinates), dtype=np.int), []
  boxes = (np.array(boxes) - vocab_start) / num_bin
  if image_size is not None:
    assert n_coordinates % 2 == 0
    h, w = image_size[:2]
    factor = [h, w] * (n_coordinates//2)
    boxes = boxes * np.expand_dims(np.array(factor), 0)

  return boxes, class_names


def build_target_image(image, image_info, crop="scale", gray_scale=False,
                       resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, to_int=False):
  """Build an image from `image` that is the same size as the original input image, before
  any scaling and cropping was done."""

  image = (image + 1) / 2.0  # Undo pre-processings
  image = tf.clip_by_value(image, 0, 1)  # We can (rarely) get negative pixel values, clip them here
  if gray_scale:
    image = tf.reduce_mean(image, -1, keepdims=True)

  off_x = int(image_info[7])
  off_y = int(image_info[8])
  src_h = int(image_info[3])
  src_w = int(image_info[4])
  h, w = image.shape[:2]
  assert h == w

  if crop == "scale" or (off_x == 0 and off_y == 0):
    if src_h > src_w:
        image = tf.image.resize(image, [src_h, src_h], method=resize_method)
    else:
        image = tf.image.resize(image, [src_w, src_w], method=resize_method)
    image = image[:src_h, :src_w]
  else:
    raise NotImplementedError()

  if to_int:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  return image.numpy()


def build_depth_prediction(image, image_info, max_depth,
                           resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
  # TODO maybe there is a better resize method for depth?
  image = build_target_image(image, image_info, gray_scale=True, crop="scale",
                             resize_method=resize_method)
  return image * max_depth
