import abc
import dataclasses
import json
import logging
import math
from os.path import join

import seqio
import numpy as np
import tensorflow as tf
# import torch
from tensorflow.python.ops import control_flow_ops

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, List
from seqio import autoregressive_inputs, non_padding_position, Vocabulary, utils, dataset_providers
from seqio.feature_converters import _check_exact_match, _check_lengths

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100 + 1000  # 1000 is for location embedding.
PAD_ID = 0


def get_default_vocabulary():
  return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.
  Subclasses of FeatureConverter are used to convert the tf.data.Dataset
  instance from the Task API to features that are passed to the
  model implementation. Note that Task API has an attribute "output_features",
  which is referred to as "model features" in the context of FeatureConverter.
  Typically the task features contain keys: "inputs" and "targets". The model
  features are constructed based on what is consumed by the model architecture.
  For custom model architectures that require additional model features, one
  needs to subclass FeatureConverter.
  This conversion is fully specified by
    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of input and output
       features in `get_model_feature_lengths` which is a function of
       task_feature_lengths.
  Therefore, a subclass of FeatureConverter should override `_convert_features`
  and `get_model_feature_lengths` methods.
  The actual feature conversion is done in the `__call__` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See `_validate_dataset` and `__call__` methods
  for more details.
  Other notes:
    If pack = True, each feature in the task features should be packable,
    i.e., 1-dimensional.
    Subclasses must override TASK_FEATURES and MODEL_FEATURES. If packing is
    used, they must override PACKING_FEATURE_DTYPES as well. These are the
    packing-specific features such as "*_segment_ids".
  Attributes:
    pack: whether to pack the dataset.
    use_custom_packing_ops: whether to use custom ops for packing.
  """

  @dataclasses.dataclass(frozen=True)
  class FeatureSpec:
    """Rank and dtype specifications for features."""
    dtype: tf.dtypes.DType
    rank: int = 1
    sequence_dim: int = 0

  TASK_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  MODEL_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  PACKING_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]

  def __init__(self,
               pack: bool = True,
               use_custom_packing_ops: bool = False):
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops

    if self.TASK_FEATURES is None:
      raise ValueError("TASK_FEATURES must be defined in the subclass.")

    if self.MODEL_FEATURES is None:
      raise ValueError("MODEL_FEATURES must be defined in the subclass.")

    if self.pack and self.PACKING_FEATURE_DTYPES is None:
      raise ValueError(
        "PACKING_FEATURE_DTYPES must be defined in the subclass if pack=True."
      )

  def _validate_dataset(
      self,
      ds: tf.data.Dataset,
      expected_features: Mapping[str, "FeatureConverter.FeatureSpec"],
      expected_lengths: Mapping[str, int],
      strict: bool,
      error_label: str) -> tf.data.Dataset:
    """Validate properties of the dataset, raising Exceptions if needed.
    This method is used to validate whether the input dataset is compatible
    with the desired specifications. In particular, the following aspects are
    checked.
    Each feature in `expected_features`
      - is also in `ds`
      - is also in expected_lengths
      - is compatible with the expected lengths
    The compatibility of the length is controlled by strict. If true, the length
    of each feature should exactly match the expected length whereas false
    condition allows the length to be less than or equal to the expected length.
    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features
      expected_lengths: a mapping from feature to its length
      strict: whether the lengths should be strictly equal or a length less than
        or equal to expected length is allowed.
      error_label: a label used to indicate the validation stage
    Returns:
      ds: the same dataset as but with the assertion ops attached.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError("Dataset is missing an expected feature during "
                         f"{error_label} validation: '{feat}'")

      if expected_features[feat].dtype != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
          f"Dataset has incorrect type for feature '{feat}' during "
          f"{error_label} validation: Got {actual_dtype}, expected "
          f"{expected_features[feat].dtype.name}")

      if expected_features[feat].rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
          f"Dataset has incorrect rank for feature '{feat}' during "
          f"{error_label} validation: "
          f"Got {actual_rank}, expected {expected_features[feat].rank}")

    sequence_axis_mapping = {
      feat: expected_features[feat].sequence_dim for feat in expected_features
    }
    ds = _check_lengths(ds, expected_lengths, sequence_axis_mapping, strict,
                        error_label)
    return ds

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    r"""Convert the features of `ds` into output features.
    This method should not be overridden by subclasses.
    There are two conversion steps and five validation steps.
    Conversion 1: task features are converted to model features in
                  `_convert_features
    Conversion 2: task_feature_lengths are converted to model_feature_lengths in
                  `get_model_feature_lengths`
    Validation 1: verifies that the user input `task_feature_lengths` only
                  contains the required features.
    Validation 2: verifies whether the input dataset has same or more features,
                  same dtype, and length that is less than or equal compared to
                  input_ds.
    Validation 3: partially verifies the behavior of overridden
                  `get_model_feature_lengths`.
    Validation 4: check whether the output dataset has expected features (extra
                  features are allowed), dtype, rank and lengths (exact match).
    Validation 5: check one-to-one match between the output dataset and
                  `expected_dtypes`. Extra features are not allowed.
    The following diagram describes the validation and conversion processes. We
    treat features in the TASK_FEATURES and MODEL_FEATURES specified as class
    variables as the ground-truth. For validations 3, 4 and 5, we define
    `expected_dtypes`.
    There are 5 validation steps. features (<=) means that features of the
    variable on the left is a subset of those of the variable on the right. For
    example, validation 2 guarantees that TASK_FEATURES has features that are a
    subset of the features of input_ds. Validation 4 has length (==), which
    means that it ensures that each feature in MODEL_FEATURES has the same
    length as the corresponding feature in output_ds.
    Overall, these 5 validations ensures that the output_ds has the expected
    features with exact length, dtype and rank. Again, these validations assume
    that TASK_FEATURES and MODEL_FEATURES are correct.
                        Validation 1                     Validation 2
    task_feature_lengths <-----------> TASK_FEATURES <----------------> input_ds
    |                   features (==)                    features (<=)        |
    |                                                    dtype (==)           |
    |                                                    length (<=)          |
    |                                                    rank (==1)           |
    |                                                                         |
    |   Conversion 2                                           Conversion 1   |
    | get_model_feature_lengths                             _convert_features |
    |                                                                         |
    |                                              Validation 5               |
    |                                           <-------------------->        |
    |                                                 features (==)           |
    |                                                                         |
    \/                    Validation 3                    Validation 4        \/
    model_feature_lengths <-------> expected_dtypes <----------------> output_ds
                        features (==)                     features (<=)
                                                          dtype (==)
                                                          length (==)
                                                          rank (==1)
    Args:
      ds: a tf.data.Dataset to be validated
      task_feature_lengths: a mapping from a task feature to its length
    Returns:
      ds: the converted dataset.
    """

    # This handles a case where seqio.evaluation tries to helpfully auto-infers the
    # max feature lengths, but doesn't understand that TASK_FEATURES might
    # be a sub-set of the dataset features and therefore gives us max feature for
    # features that are passed through and not packed by the converter, which in
    # turns fails the validation check. We fix that here to avoid having to messing with seqio
    # interals
    task_feature_lengths = {k: v for k, v in task_feature_lengths.items() if k in self.TASK_FEATURES}

    # Validation 1
    _check_exact_match(expected_features=list(self.TASK_FEATURES),
                       actual_features=list(task_feature_lengths),
                       expected_feature_source="TASK_FEATURES",
                       actual_feature_source="task_feature_lengths")

    # Validation 2
    ds = self._validate_dataset(
      ds,
      expected_features=self.TASK_FEATURES,
      expected_lengths=task_feature_lengths,
      # Before pack/pad, check feature (of ds) length <= task feature length
      strict=False,
      error_label="input_validation")

    # Conversion 1: implemented by subclass
    ds = self._convert_features(ds, task_feature_lengths)

    expected_features = dict(self.MODEL_FEATURES)
    if self.pack:
      for k, v in expected_features.items():
        # Packing requires rank 1.
        if v.rank != 1 and not self._use_custom_packing_ops:
          raise ValueError(
            f"When packing is enabled, expected ranks must be 1 or "
            f"use_custom_packing_ops must be set. Got expected rank {v.rank} "
            f"for feature {k}.")
      for k, v in self.PACKING_FEATURE_DTYPES.items():
        expected_features[k] = FeatureConverter.FeatureSpec(rank=1, dtype=v)

    # Conversion 2: implemented by subclasses
    model_feature_lengths = self.get_model_feature_lengths(task_feature_lengths)
    # Validation 3

    _check_exact_match(expected_features=list(expected_features),
                       actual_features=list(model_feature_lengths),
                       expected_feature_source="model_feature_names",
                       actual_feature_source="model_feature_lengths")

    # Validation 4
    ds = self._validate_dataset(
      ds,
      expected_features=expected_features,
      expected_lengths=model_feature_lengths,
      # After pack/pad, check feature (of ds) length == model feature length
      strict=True,
      error_label="output_validation")

    # # Validation 5
    # _check_exact_match(expected_features=list(expected_features),
    #                    actual_features=list(ds.element_spec),
    #                    expected_feature_source="model_feature_names",
    #                    actual_feature_source="output_dataset")
    return ds

  def _pack_or_pad(self,
                   ds: tf.data.Dataset,
                   packed_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Trim/pad to packed_lengths and optionally pack the input dataset."""
    if self.pack:
      ds = utils.trim_and_pad_dataset(ds, packed_lengths)
    else:
      ds = utils.trim_and_pad_dataset(ds, packed_lengths)
    return ds

  @abc.abstractmethod
  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def pack(self) -> bool:
    return self._pack


class UnifiedIOFeatureConverter(FeatureConverter):
  """
  """
  TASK_FEATURES = {
    "text_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
    "text_encoder_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_encoder_masks": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_masks": FeatureConverter.FeatureSpec(dtype=tf.float32),
  }
  PACKING_FEATURE_DTYPES = {
    "text_decoder_segment_ids": tf.int32,
    "text_decoder_positions": tf.int32,
  }

  def __init__(
      self, pack: bool = True, use_custom_packing_ops: bool = False, pass_through=None):
    super().__init__(pack, use_custom_packing_ops)
    self.pass_through = pass_through

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.
    The conversion process involves three steps
    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.
    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.
    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.
    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:

      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = autoregressive_inputs(
        features["text_targets"],
        sequence_id=features.get("text_decoder_segment_ids", None))

      text_decoder_masks = non_padding_position(features['text_targets'], dtype=tf.float32) * features[
        'score'] if 'score' in features \
        else non_padding_position(features['text_targets'], dtype=tf.float32)

      d = {"text_encoder_inputs": features["text_inputs"],
           "text_encoder_masks": non_padding_position(features["text_inputs"]),
           "text_decoder_targets": features["text_targets"],
           "text_decoder_inputs": decoder_input_tokens,
           # Loss is computed for all but the padding positions.
           "text_decoder_masks": text_decoder_masks,
           "image_encoder_inputs": features["image_inputs"],
           "image_decoder_targets": features["image_targets"],
           "image_input_masks": features["image_input_masks"],
           }

      optional_features = [
        "image_target_masks",
        'image_target_loss_masks',
        'image_encoder_pos_ids',
        'text_encoder_pos_ids',
        'text_decoder_positions',
        'output_options',
        'num_turns',
      ]
      for k in optional_features:
        if k in features:
          d[k] = features[k]

      if self.pass_through:
        for k in self.pass_through:
          d[k] = features[k]

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
      convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    text_encoder_length = task_feature_lengths["text_inputs"]
    text_decoder_length = task_feature_lengths["text_targets"]

    model_feature_lengths = {
      "text_encoder_inputs": text_encoder_length,
      "text_encoder_masks": text_encoder_length,
      "text_decoder_targets": text_decoder_length,
      "text_decoder_inputs": text_decoder_length,
      "text_decoder_masks": text_decoder_length
    }
    if self.pack:
      model_feature_lengths["text_decoder_segment_ids"] = text_decoder_length
      model_feature_lengths["text_decoder_positions"] = text_decoder_length

    return model_feature_lengths


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


def unnormalize_image(image,
                      offset=(0.485, 0.456, 0.406),
                      scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  scale = tf.expand_dims(tf.expand_dims(tf.constant(scale), axis=0), axis=0)
  image *= scale

  offset = tf.expand_dims(tf.expand_dims(tf.constant(offset), axis=0), axis=0)
  image += offset
  return image


def flip_if_vertical(image):
  """
    https://www.youtube.com/watch?v=f2picMQC-9E
    :param image:
    :return:
    """
  height = tf.cast(tf.shape(image)[0], tf.float32)
  width = tf.cast(tf.shape(image)[1], tf.float32)
  # Pad and then add some constants (if it's flipped) to tell the model that we messed with it
  image = tf.cond(
    height >= (4 * width / 3.0),
    lambda: tf.pad(tf.image.rot90(image), [[0, 0], [4, 4], [0, 0]], mode='CONSTANT', constant_values=0.5),
    lambda: image,
  )
  return image


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
    func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
    for case in range(num_cases)])[0]


def denormalize_boxes(boxes, image_shape):
  """Converts boxes normalized by [height, width] to pixel coordinates.
    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates of
        boxes in ymin, xmin, ymax, xmax order.
      image_shape: a list of two integers, a two-element vector or a tensor such
        that all but the last dimensions are `broadcastable` to `boxes`. The last
        dimension is 2, which represents [height, width].
    Returns:
      denormalized_boxes: a tensor whose shape is the same as `boxes` representing
        the denormalized boxes.
    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
  with tf.name_scope('denormalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.split(image_shape, 2, axis=-1)

    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width

    denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
      boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height - 1.0, width - 1.0, height - 1.0, width - 1.0]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
        [height - 1.0, width - 1.0, height - 1.0, width - 1.0], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def get_non_empty_box_indices(boxes):
  """Get indices for non-empty boxes."""
  # Selects indices if box height or width is 0.
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  indices = tf.where(
    tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
  return indices[:, 0]


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
  """Resizes boxes to output size with scale and offset.
    Args:
      boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
      image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
      output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
      offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.
    Returns:
      boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
  # Adjusts box coordinates based on image_scale and offset.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
  # Clips the boxes.
  boxes = clip_boxes(boxes, output_size)
  return boxes


def resize_and_pad(image, desired_output_size, masks=None, boxes=None, labels=None,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True, boxes1=None, filter_box=True,
                   do_flip_if_vertical=True, desired_target_size=None, random_scale_ratio=0.0,
                   resize_method=tf.image.ResizeMethod.BILINEAR):
  """
    :param image:
    :param desired_output_size:
    :param boxes:
    :param random_scale_min:
    :param random_scale_max:
    :param do_random_scale:
    :param shrink_both_sides: whether both sides can be shrunk at the same time
    :return:
    """
  if do_flip_if_vertical:
    image = flip_if_vertical(image)

  desired_height, desired_width = desired_output_size
  desired_height_f = tf.cast(desired_height, dtype=tf.float32)
  desired_width_f = tf.cast(desired_width, dtype=tf.float32)

  height = tf.cast(tf.shape(image)[0], tf.float32)
  width = tf.cast(tf.shape(image)[1], tf.float32)

  if boxes is not None:
    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = denormalize_boxes(boxes, [height, width])

  if boxes1 is not None:
    boxes1 = denormalize_boxes(boxes1, [height, width])

  if do_random_scale:
    random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
    if not shrink_both_sides:
      # Max random is where scale * W > W_desired
      #                     scale * H > H_desired
      rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
      random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

    scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
    scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

    # Recompute the accurate scale_factor using rounded scaled image size.
    image_scale_y = tf.cast(scaled_y, tf.float32) / height
    image_scale_x = tf.cast(scaled_x, tf.float32) / width

    image_scale = tf.cond(tf.less(
      tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
      tf.cast(random_scale_ratio, tf.float32)),
      lambda: tf.maximum(image_scale_x, image_scale_y),
      lambda: tf.minimum(image_scale_x, image_scale_y))

    # image_scale = tf.minimum(image_scale_x, image_scale_y)

    # Conceptual captions has some REALLY WIDE images I believe
    # this ensures that we won't scale any side lower than to 64
    image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    offset_y = tf.cast(scaled_height - desired_height, tf.float32)
    offset_x = tf.cast(scaled_width - desired_width, tf.float32)
    offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
    offset_y = tf.cast(offset_y, tf.int32)
    offset_x = tf.cast(offset_x, tf.int32)
  else:
    image_scale_y = desired_height_f / height
    image_scale_x = desired_width_f / width
    image_scale = tf.minimum(image_scale_x, image_scale_y)
    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    offset_y = tf.constant(0)
    offset_x = tf.constant(0)

  # Now resize and crop
  if resize_method == 'random' and do_random_scale and (not tf.executing_eagerly()):
    resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
    # print("Random resize method:\n{}".format(','.join(resize_methods)))
    image = apply_with_random_selector(
      image,
      lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                            tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                            antialias=True),
      num_cases=len(resize_methods))

  elif resize_method != 'random':
    image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
  else:
    print(f"you passed in {resize_method} but doing bilinear resize instead (possibly because eager is on)")
    image = tf.image.resize(image, [scaled_height, scaled_width],
                            method=tf.image.ResizeMethod.BILINEAR, antialias=True)

  image = tf.clip_by_value(image, 0.0, 1.0)
  image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
  image_mask = tf.image.pad_to_bounding_box(
    tf.ones_like(image), 0, 0, desired_height, desired_width)[:, :, 0]
  image = tf.image.pad_to_bounding_box(image, 0, 0, desired_height, desired_width)

  if isinstance(desired_height, int) and isinstance(desired_width, int):
    image.set_shape([desired_height, desired_width, 3])
  else:
    print("Cant set shape bc desired height/width are dynamic")

  if masks is not None and tf.size(masks) != 0:
    masks = tf.image.resize(masks, [scaled_height, scaled_width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if len(masks.shape) == 3:
      masks = masks[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]
    else:
      masks = masks[:, offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]

    masks = tf.image.pad_to_bounding_box(masks, 0, 0, desired_height, desired_width)
    masks = tf.image.resize(masks, desired_target_size,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  indices = None
  if boxes is not None:
    boxes = resize_and_crop_boxes(
      boxes,
      tf.stack([image_scale, image_scale]),
      [desired_height, desired_width],
      tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32))

    if filter_box:
      indices = get_non_empty_box_indices(boxes)
    else:
      indices = tf.range(tf.shape(boxes)[0])
    boxes = tf.gather(boxes, indices)

    if labels is not None:
      labels = tf.gather(labels, indices)

    if boxes1 is not None:
      boxes1 = resize_and_crop_boxes(
        boxes1,
        tf.stack([image_scale, image_scale]),
        [desired_height, desired_width],
        tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32))

  effective_height = tf.minimum(scaled_height, desired_height)
  effective_width = tf.minimum(scaled_width, desired_width)

  image_info = tf.stack([
    tf.cast(effective_height, dtype=tf.float32) / desired_height_f,
    tf.cast(effective_width, dtype=tf.float32) / desired_width_f,
    1.0 / image_scale,
    height,
    width,
    tf.cast(offset_y, dtype=tf.float32) / height,
    tf.cast(offset_x, dtype=tf.float32) / width,
    tf.cast(offset_y, dtype=tf.float32),
    tf.cast(offset_x, dtype=tf.float32),
    tf.cast(scaled_height, dtype=tf.float32),
    tf.cast(scaled_width, dtype=tf.float32),
  ])

  if boxes1 is not None:
    outputs = (image_info, masks, boxes, labels, indices, boxes1)
  else:
    outputs = (image_info, masks, boxes, labels, indices)

  return image, image_mask, outputs


def convert_keypoint_to_sequence(
    boxes,
    labels,
    num_bin,
    image_size,
    vocab_start):
  # shuffle the labels, ids.
  quantized_boxes = tf.cast((num_bin - 1) / image_size * boxes, tf.int32)
  vals_tensor = tf.constant([f'<extra_id_{i}>' for i in range(vocab_start, vocab_start + num_bin)])
  keys_tensor = tf.constant([i for i in range(num_bin)])
  table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
      keys_tensor,
      vals_tensor
    ),
    default_value='<extra_id_0>'
  )

  # we use the <extra_id_99> to represent the special tokens.
  quantized_boxes_all = quantized_boxes + tf.cast((quantized_boxes == 0), tf.int32) * num_bin
  boxes_str = table.lookup(quantized_boxes_all)

  label_vals_tensor = tf.constant([f'{i + 1}' for i in range(3)])
  label_keys_tensor = tf.constant([i for i in range(3)])
  class_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
      label_keys_tensor,
      label_vals_tensor
    ),
    default_value='1'
  )
  labels = tf.cast(labels, tf.int32)
  labels_str = class_table.lookup(labels)

  output_str = tf.concat([boxes_str, labels_str], axis=1)
  output_str = tf.reshape(output_str, [-1])
  output_str = tf.strings.reduce_join(output_str, separator=' ')
  return output_str


def generate_random_box(
    all_names,
):
  rand_label = tf.random.uniform([], minval=0, maxval=len(all_names), dtype=tf.int32)
  rand_name = tf.constant(all_names)[rand_label]
  return tf.constant(''), rand_name


def convert_bbox_to_sequence(
    boxes,
    labels,
    label_names,
    num_bin,
    image_size,
    vocab_start,
    maximum_num_boxes=75,
    label_drop=0.5,
    scale_min=0.8,
    scale_max=1.2,
    shift_min=-0.2,
    shift_max=0.2,
    convert_to_str=True,
    shuffle=True,
    concat_label_str=True,
):
  if shuffle:
    # shuffle the labels, ids.
    indices = tf.range(start=0, limit=tf.shape(boxes)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    if labels is not None:
      labels = tf.gather(labels, shuffled_indices)
    boxes = tf.gather(boxes, shuffled_indices)

  quantized_boxes = tf.cast((num_bin - 1) / image_size * boxes, tf.int32)

  vals_tensor = tf.constant([f'<extra_id_{i}>' for i in range(vocab_start, vocab_start + num_bin)])
  keys_tensor = tf.constant([i for i in range(num_bin)])
  table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
      keys_tensor,
      vals_tensor
    ),
    default_value='<extra_id_0>'
  )
  # add negative padding.
  # half of the them are random box.
  # num_pos = tf.shape(quantized_boxes)[0]
  # num_noise = maximum_num_boxes - num_pos

  # if num_pos > 0:
  #   num_sc_noise = num_noise // 2
  #   rand_scale = scale_min + tf.random.uniform([num_sc_noise, 1]) * (scale_max - scale_min)

  #   # random shifting (-0.2 - 0.2)
  #   rand_shift = shift_min + tf.random.uniform([num_sc_noise, 2]) * (shift_max - shift_min)

  #   # replicating existing boxes.
  #   replicate_boxes = tf.tile(boxes, [num_sc_noise//num_pos+1, 1])[:num_sc_noise]
  #   replicate_labels = tf.tile(labels, [num_sc_noise//num_pos+1])[:num_sc_noise]

  #   # applying the random scaling and shifting.
  #   sc_boxes = replicate_boxes * rand_scale

  #   sc_boxes_x1 = sc_boxes[:,0:1] + rand_shift[:,0:1]
  #   sc_boxes_x2 = sc_boxes[:,2:3] + rand_shift[:,0:1]
  #   sc_boxes_y1 = sc_boxes[:,1:2] + rand_shift[:,1:2]
  #   sc_boxes_y2 = sc_boxes[:,3:4] + rand_shift[:,1:2]
  #   sc_boxes = tf.concat([sc_boxes_x1, sc_boxes_y1, sc_boxes_x2, sc_boxes_y2], axis=1)

  #   # make sure the sc_box in the correct range.
  #   sc_boxes = tf.clip_by_value(sc_boxes, 0, image_size)
  #   sc_boxes = tf.cast((num_bin-1) / image_size * sc_boxes, tf.int32)
  # else:
  #   num_sc_noise = 0
  #   sc_boxes = tf.reshape(tf.convert_to_tensor(()), (0, 4))
  #   sc_boxes = tf.cast(sc_boxes, tf.int32)
  #   replicate_labels = tf.reshape(tf.convert_to_tensor(()), (0, ))
  #   replicate_labels = tf.cast(replicate_labels, tf.int32)

  # # half of them are jittering box.
  # num_random_noise = num_noise - num_sc_noise
  # rand_xy =  tf.random.uniform([num_random_noise, 2])
  # rand_wh = tf.random.uniform([num_random_noise, 2]) * (1-rand_xy)
  # rand_boxes = tf.concat([rand_xy, rand_xy+rand_wh], axis=1) * image_size
  # rand_boxes = tf.cast((num_bin-1) / image_size * rand_boxes, tf.int32)
  # rand_label = tf.cast(tf.random.uniform([num_random_noise])*len(label_names), tf.int32)

  # quantized_boxes_all = tf.concat([quantized_boxes, sc_boxes, rand_boxes], axis=0)
  # labels_all = tf.concat([labels, replicate_labels, rand_label], axis=0)

  quantized_boxes_all = quantized_boxes
  boxes_str = table.lookup(quantized_boxes_all)
  if labels is not None:
    # convert labels to string.
    if label_names is not None:
      labels = tf.cast(labels, tf.int32)
      class_vals_tensor = tf.constant(label_names)
      class_keys_tensor = tf.constant([i for i in range(len(label_names))])
      class_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
          class_keys_tensor,
          class_vals_tensor
        ),
        default_value='none'
      )
      labels_str = class_table.lookup(labels)
    else:
      labels_str = labels

    labels_str = tf.expand_dims(labels_str, axis=-1)
    if concat_label_str:
      class_label_str = tf.concat([boxes_str, labels_str], axis=-1)
    else:
      class_label_str = boxes_str

    if convert_to_str:
      class_label_str = tf.reshape(class_label_str, [-1])
      class_label_str = tf.strings.reduce_join(class_label_str, separator=' ')

    if label_names is not None:
      unique_labels, _ = tf.unique(labels)
      unique_labels_str = class_table.lookup(unique_labels)
      unique_labels_str = tf.reshape(unique_labels_str, [-1])
      unique_labels_str = tf.strings.reduce_join(unique_labels_str, separator=' ')
    else:
      unique_labels_str, _ = tf.unique(labels)
      unique_labels_str = tf.strings.reduce_join(unique_labels_str, separator=' ')
  else:
    class_label_str = tf.reshape(boxes_str, [-1])
    class_label_str = tf.strings.reduce_join(class_label_str, separator=' ')
    unique_labels_str = None

  return class_label_str, unique_labels_str


OTHER_INSTANCE_COLORS = [
  [0, 0, 127],
  [0, 0, 255],
  [0, 127, 0],
  [0, 255, 0],
  [127, 0, 0],
  [255, 0, 0],
]
FIRST_OBJ_COLOR = [255, 255, 255]
BK_COLORS = [0, 0, 0]


def convert_panoptic_image_to_rgb(panoptic_image, ids):
  '''
    Convert to panoptic segmentation to quantized rgb space.
    This function is written for the tfds coco panoptic dataset.
    '''
  color_dict = tf.constant(OTHER_INSTANCE_COLORS, dtype=tf.uint8)
  color_dict = tf.random.shuffle(color_dict)
  color_dict = tf.concat([
    tf.constant([BK_COLORS], dtype=tf.uint8),
    tf.constant([FIRST_OBJ_COLOR], dtype=tf.uint8),
    color_dict,
  ], axis=0)

  panoptic_image = tf.cast(panoptic_image, dtype=tf.int32)
  # [h, w], the ids of each pixel
  panoptic_image_sum = panoptic_image[:, :, 0] + panoptic_image[:, :, 1] * 256 + panoptic_image[:, :, 2] * 256 * 256

  ids = tf.cast(ids, dtype=tf.int32)
  n_instance_ids = len(OTHER_INSTANCE_COLORS) + 1
  tf.random.shuffle(ids)  # Shuffle ids so if we have to skip instances, we skip random ones
  ids = ids[:n_instance_ids]  # Throw out instances we don't have color for

  matches = tf.expand_dims(panoptic_image_sum, -1) == tf.reshape(ids, (1, 1, -1))

  # TODO can we use boolean_mask instead? This throws an index error
  # matches = tf.concat([tf.reduce_any(matches, axis=-1, keepdims=True), matches], -1)
  # img = tf.boolean_mask(tf.expand_dims(tf.expand_dims(color_dict[:tf.shape(ids)[-1]+1], 0), 0), matches)
  # img = tf.reshape(img, tf.shape(panoptic_image_sum))

  ixs = tf.constant(np.arange(1, color_dict.shape[0] + 1, dtype=np.int32))[:tf.shape(ids)[0]]
  # [h, w] the index of the id each pixel matches, or zero if no matches
  ixs = tf.reduce_sum(tf.cast(matches, ids.dtype) * tf.reshape(ixs, (1, 1, -1)), -1)

  # [h, w] color corresponding to each id
  img = tf.gather(color_dict, ixs)

  return tf.image.convert_image_dtype(img, dtype=tf.float32)


def convert_segmentation_to_rgb(segmentation, labels, label_names):
  segmentation = tf.squeeze(segmentation, axis=-1)
  color_dict = tf.constant([BK_COLORS, FIRST_OBJ_COLOR], dtype=tf.uint8)
  color_dict_2 = tf.constant(OTHER_INSTANCE_COLORS, dtype=tf.uint8)
  color_dict_2 = tf.random.shuffle(color_dict_2)
  color_dict = tf.concat([color_dict, color_dict_2], axis=0)
  if label_names is not None:
    labels = tf.cast(labels, dtype=tf.int32)
  # random select an class we want to do segmentation.
  # TODO no longer need this since we flatten by label
  unique_labels, _ = tf.unique(labels)
  rand_idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(unique_labels)[0], dtype=tf.int32)
  select_label = unique_labels[rand_idx]

  select_mask = (labels == select_label)
  select_indices = tf.reshape(tf.where(select_mask), (-1,))
  select_indices = tf.cast(select_indices, tf.int32)
  select_indices = tf.random.shuffle(select_indices)[:7]

  segmentation = tf.gather(segmentation, select_indices)

  segmentation = tf.cast(segmentation / 255, tf.int32)

  segmentation_convert = segmentation * tf.reshape(tf.range(tf.shape(select_indices)[0]) + 1, [-1, 1, 1])
  segmentation_convert = tf.math.reduce_max(segmentation_convert, axis=0)

  panoptic_image_rgb = tf.gather(color_dict, segmentation_convert)
  panoptic_image_rgb = tf.image.convert_image_dtype(panoptic_image_rgb, dtype=tf.float32)

  return panoptic_image_rgb, segmentation


GEN_SEGMENTATION_COLORS = tf.convert_to_tensor([
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [255, 255, 255],
  [128, 128, 128],
  [255, 0, 255],
  [255, 255, 0],
  [0, 255, 255],
  [192, 192, 192],
  [128, 0, 0],
  [128, 128, 0],
  [0, 128, 0],
  [0, 128, 128],
  [0, 0, 128],
  [128, 0, 128],
], dtype=tf.uint8)

GEN_SEGMENTATION_COLOR_NAMES = tf.constant([
  'red',
  'lime',
  'blue',
  'white',
  'gray',
  'fuchsia',
  'yellow',
  'aqua',
  'silver',
  'maroon',
  'olive',
  'green',
  'teal',
  'navy',
  'purple'
])


def get_segmentation_colors():
  indices = tf.range(start=0, limit=tf.shape(GEN_SEGMENTATION_COLORS)[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  color = tf.gather(GEN_SEGMENTATION_COLORS, shuffled_indices)
  names = tf.gather(GEN_SEGMENTATION_COLOR_NAMES, shuffled_indices)
  names = tf.concat([
    ["black"],
    names
  ], 0)
  color = tf.concat([
    tf.convert_to_tensor([[0, 0, 0]], dtype=tf.uint8),
    color
  ], 0)
  return names, color


def shuffle_tensors(*args):
  indices = tf.range(start=0, limit=tf.shape(args[0])[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  return [tf.gather(x, shuffled_indices) for x in args]


def convert_panoptic_image_to_rgb_semantic(panoptic_image, ids, labels, label_names):
  '''
    Convert to panoptic segmentation to quantized rgb space.
    This function is written for the tfds coco panoptic dataset.
    '''
  names, colors = get_segmentation_colors()

  panoptic_image = tf.cast(panoptic_image, dtype=tf.int32)
  panoptic_image_sum = panoptic_image[:, :, 0] + panoptic_image[:, :, 1] * 256 + panoptic_image[:, :, 2] * 256 * 256
  panoptic_image_sum = tf.reshape(panoptic_image_sum, (-1,))

  # panoptic_image_ids: unique id for each panoptic sum in the image
  # ids_to_sum: the panoptic sum those ids correspond to
  panoptic_sums, panoptic_sum_ids = tf.unique(panoptic_image_sum)

  # Maps to panoptic_sums to `instance_ids`, the index in ids/labels plus 1 of that sum, or
  # zero for background pixels
  # We allow panoptic sums that do not exist in `ids` to become background (zero)
  ids_with_bg = tf.pad(ids, tf.constant([[1, 0]]))
  matches = tf.expand_dims(tf.cast(panoptic_sums, ids_with_bg.dtype), 1) == tf.expand_dims(ids_with_bg, 0)
  ixs = tf.expand_dims(tf.range(0, tf.shape(matches)[1], dtype=tf.int32), 0)
  instance_ids = tf.reduce_max(tf.cast(matches, tf.int32) * ixs, -1)

  # Convert to labels to `labels_id` tensor that maps each label to a sequential id, with
  # zero as the background class
  unique_labels, label_ids = tf.unique(labels)
  label_ids = label_ids + 1
  label_ids = label_ids * tf.cast(label_ids <= 15, label_ids.dtype)  # ids>=15 are treated as background
  unique_labels = unique_labels[:15]
  label_ids = tf.pad(label_ids, tf.constant([[1, 0]]))

  # panoptic_sum_ids -> instance_id -> label_id -> color
  image_instance_ids = tf.gather(instance_ids, panoptic_sum_ids, axis=0)
  image_label_ids = tf.gather(label_ids, image_instance_ids, axis=0)
  panoptic_image_rgb = tf.gather(colors, image_label_ids, axis=0)
  panoptic_image_rgb = tf.reshape(panoptic_image_rgb, tf.shape(panoptic_image))
  panoptic_image_rgb = tf.image.convert_image_dtype(panoptic_image_rgb, dtype=tf.float32)

  # `unique_labels` holds the labels we used in the image, convert to the prompt
  text_labels_in_image = tf.gather(label_names, unique_labels)
  colors_in_image = names[1:tf.shape(unique_labels)[0] + 1]  # to skip bg color
  input_str = tf.stack([colors_in_image, text_labels_in_image], axis=1)
  input_str = tf.strings.reduce_join(input_str, axis=-1, separator=" : ")
  input_str = tf.strings.reduce_join(input_str, axis=-1, separator=" , ")
  return panoptic_image_rgb, input_str


def convert_segmentation_to_rgb_semantic(segmentation, labels, label_names):
  names, colors = get_segmentation_colors()

  segmentation = tf.squeeze(segmentation, axis=-1)

  unique_labels, label_ids = tf.unique(labels)
  label_ids = label_ids + 1
  label_ids = label_ids * tf.cast(label_ids <= 15, label_ids.dtype)  # ids>=15 are treated as background
  unique_labels = unique_labels[:15]
  label_ids = tf.pad(label_ids, tf.constant([[1, 0]]))

  segmentation = tf.cast(segmentation / 255, tf.int32)
  segmentation_convert = segmentation * tf.reshape(tf.range(tf.shape(labels)[0]) + 1, [-1, 1, 1])
  segmentation_convert = tf.math.reduce_max(segmentation_convert, axis=0)

  image_label_ids = tf.gather(label_ids, segmentation_convert)
  panoptic_image_rgb = tf.gather(colors, image_label_ids)
  panoptic_image_rgb = tf.image.convert_image_dtype(panoptic_image_rgb, dtype=tf.float32)

  text_labels_in_image = tf.gather(label_names, unique_labels)
  colors_in_image = names[1:tf.shape(unique_labels)[0] + 1]  # to skip bg color
  input_str = tf.stack([colors_in_image, text_labels_in_image], axis=1)
  input_str = tf.strings.reduce_join(input_str, axis=-1, separator=" : ")
  input_str = tf.strings.reduce_join(input_str, axis=-1, separator=" , ")
  return panoptic_image_rgb, input_str


def plot_bbox(image, boxes):
  from torchvision.utils import draw_bounding_boxes, save_image
  import torch
  boxes_plot = torch.tensor(boxes.numpy())
  image_plot = torch.tensor(image.numpy() * 255, dtype=torch.uint8)
  image_plot = torch.permute(image_plot, (2, 0, 1))
  plot = draw_bounding_boxes(image_plot,
                             torch.stack([boxes_plot[:, 1], boxes_plot[:, 0], boxes_plot[:, 3], boxes_plot[:, 2]],
                                         dim=1))
  save_image(plot.float() / 255, 'test.jpg')


def encode_multi_text_targets(inputs, vocab, target_length):
  eos_id = vocab.eos_id
  inputs = seqio.preprocessors._append_to_innermost_axis(inputs, eos_id)
  row_length = inputs.row_lengths()
  position_ids = tf.cast(tf.ragged.range(row_length).flat_values, tf.int32)
  segment_ids = tf.cast(inputs.value_rowids() + 1, tf.int32)
  inputs = tf.cast(inputs.flat_values, tf.int32)
  seq_len = tf.shape(segment_ids)[0]

  if seq_len < target_length:
    segment_ids = tf.concat([
      segment_ids, tf.zeros(target_length - seq_len, dtype=tf.int32)], axis=0)
    position_ids = tf.concat([
      position_ids, tf.zeros(target_length - seq_len, dtype=tf.int32)], axis=0)
  else:
    segment_ids = tf.slice(segment_ids, [0], [target_length])
    position_ids = tf.slice(position_ids, [0], [target_length])

  segment_ids = tf.reshape(segment_ids, (target_length,))
  position_ids = tf.reshape(position_ids, (target_length,))

  return inputs, segment_ids, position_ids


def encode_multi_text_targets_eval(inputs, vocab, target_length):
  eos_id = vocab.eos_id
  inputs = seqio.preprocessors._append_to_innermost_axis(inputs, eos_id)

  nrow = inputs.nrows()
  position_ids = tf.expand_dims(tf.range(target_length), axis=0)
  position_ids = tf.repeat(position_ids, nrow, axis=0)
  position_ids = tf.cast(tf.reshape(position_ids, [-1]), tf.int32)

  segment_ids = tf.expand_dims(tf.range(nrow) + 1, axis=1)
  segment_ids = tf.repeat(segment_ids, target_length, axis=1)
  segment_ids = tf.cast(tf.reshape(segment_ids, [-1]), tf.int32)

  inputs = tf.cast(inputs.to_tensor(default_value=0, shape=[nrow, target_length]), tf.int32)
  inputs = tf.reshape(inputs, [-1])
  seq_len = tf.shape(segment_ids)[0]

  return inputs, segment_ids, position_ids


def generate_mask(input_size, num_masking_patches, min_num_patches=4,
                  max_num_patches=None, min_aspect=0.3, max_aspect=None):
  max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
  max_aspect = max_aspect or 1 / min_aspect
  mask = tf.zeros([input_size, input_size], dtype=tf.int32)
  mask_count = tf.constant(0)
  try_count = tf.constant(0)

  def cond_fn(mask_count, try_count, *args):
    return tf.less(mask_count, num_masking_patches) and tf.less(try_count, 5)

  def body_fn(mask_count, try_count, mask):
    target_area = tf.random.uniform([], min_num_patches, max_num_patches)
    aspect_ratio = tf.math.exp(tf.random.uniform([], math.log(min_aspect), math.log(max_aspect)))
    h = tf.cast(tf.math.sqrt(target_area * aspect_ratio), dtype=tf.int32)
    w = tf.cast(tf.math.sqrt(target_area / aspect_ratio), dtype=tf.int32)

    if w < input_size and h < input_size:
      top = tf.random.uniform([], 0, input_size - h, dtype=tf.int32)
      left = tf.random.uniform([], 0, input_size - w, dtype=tf.int32)
      # num_masked = tf.math.reduce_sum(tf.cast(mask[top: top + h, left: left + w] !=0, dtype=tf.int32))
      # if num_masked == 0:
      y = tf.range(h) + top
      x = tf.range(w) + left
      indices = tf.reshape(tf.stack(tf.meshgrid(y, x), axis=-1), (-1, 2))
      updates = tf.ones(tf.shape(indices)[0], dtype=tf.int32)
      mask = tf.tensor_scatter_nd_update(mask, indices, updates)
      mask_count += tf.math.reduce_sum(updates)

    try_count += 1
    return mask_count, try_count, mask

  mask_count, try_count, mask = tf.while_loop(cond_fn, body_fn, [mask_count, try_count, mask])

  return mask


def box_mask(box, image_size):
  x, y = image_size
  xmin, ymin, xmax, ymax = tf.unstack(box)
  h = xmax - xmin
  z0 = tf.zeros([xmin, y])
  z1 = tf.concat(
    [tf.zeros([h, ymin]),
     tf.ones([h, ymax - ymin]),
     tf.zeros([h, y - ymax])],
    axis=1)
  z2 = tf.zeros([x - xmax, y])
  return tf.concat([z0, z1, z2], axis=0)


def load_coco_panoptic_class_names(remove_stuff_suffix=False):
  names = load_class_name('metadata/coco/coco_panoptic_class_name_2017.json')
  if remove_stuff_suffix:
    names = [x.rstrip("-stuff") for x in names]
  return names


def load_class_name(path):
  with open(path) as f:
    data = json.load(f)
  data = {int(k): v for k, v in data.items()}
  keys = sorted(data)
  if path == "metadata/imagenet/imagenet_2012_class_name.json":
    # Our meta-data for imagenet stores class names starting at 1
    assert keys == list(range(1, 1001))
  else:
    # preprocessing code assumed this sorted mapping
    assert keys == list(range(len(data)))
  return [data[k] for k in keys]


def load_answer_options(name) -> List[str]:
  src = join(f"metadata/{name}.json")
  with open(src) as f:
    options = json.load(f)
  return options


def get_answer_options_as_tokens(name):
  # This is a bit hacky since we assume this vocab is the same as the one the model is
  # using, although it should be a safe assumption in our case
  vocab = get_default_vocabulary()
  logging.info(f"Loading and tokenizing answer options {name}")
  options = load_answer_options(name)
  token_ids = []
  for name in options:
    token_ids.append(vocab.encode(name))

  seq_len = max(len(x) for x in token_ids)
  batch = len(token_ids)
  inputs = np.zeros((batch, seq_len), dtype=bool)
  for i, ids in enumerate(token_ids):
    inputs[i, :len(ids)] = ids
  return inputs


class UnifiedIOPairFeatureConverter(UnifiedIOFeatureConverter):
  """
  """
  TASK_FEATURES = {
    "text_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_targets_positive": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_targets_negative": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
    "text_encoder_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_encoder_masks": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_targets_positive": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_inputs_positive": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_masks_positive": FeatureConverter.FeatureSpec(dtype=tf.float32),
    "text_decoder_targets_negative": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_inputs_negative": FeatureConverter.FeatureSpec(dtype=tf.int32),
    "text_decoder_masks_negative": FeatureConverter.FeatureSpec(dtype=tf.float32),
  }
  PACKING_FEATURE_DTYPES = {
    "text_decoder_segment_ids_positive": tf.int32,
    "text_decoder_positions_positive": tf.int32,
    "text_decoder_segment_ids_negative": tf.int32,
    "text_decoder_positions_negative": tf.int32,
  }

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.
    The conversion process involves three steps
    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.
    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.
    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.
    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:

      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens_positive = autoregressive_inputs(
        features["text_targets_positive"],
        sequence_id=features.get("text_decoder_segment_ids_positive", None))

      decoder_input_tokens_negative = autoregressive_inputs(
        features["text_targets_negative"],
        sequence_id=features.get("text_decoder_segment_ids_negative", None))

      text_decoder_masks_positive = non_padding_position(features['text_targets_positive'], dtype=tf.float32)
      text_decoder_masks_negative = non_padding_position(features['text_targets_negative'], dtype=tf.float32)

      d = {"text_encoder_inputs": features["text_inputs"],
           "text_encoder_masks": non_padding_position(features["text_inputs"]),

           "image_encoder_inputs": features["image_inputs"],
           "image_decoder_targets": features["image_targets"],
           "image_input_masks": features["image_input_masks"],

           "text_decoder_targets_positive": features["text_targets_positive"],
           "text_decoder_inputs_positive": decoder_input_tokens_positive,
           # Loss is computed for all but the padding positions.
           "text_decoder_masks_positive": text_decoder_masks_positive,

           "text_decoder_targets_negative": features["text_targets_negative"],
           "text_decoder_inputs_negative": decoder_input_tokens_negative,
           # Loss is computed for all but the padding positions.
           "text_decoder_masks_negative": text_decoder_masks_negative,
           }

      optional_features = [
        "image_target_masks",
        'image_target_loss_masks',
        'image_encoder_pos_ids',
        'text_encoder_pos_ids',
        'text_decoder_positions',
        'output_options',
        'num_turns',
      ]
      for k in optional_features:
        if k in features:
          d[k] = features[k]

      if self.pass_through:
        for k in self.pass_through:
          d[k] = features[k]

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
      convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    text_encoder_length = task_feature_lengths["text_inputs"]
    text_decoder_length = task_feature_lengths["text_targets_positive"]

    model_feature_lengths = {
      "text_encoder_inputs": text_encoder_length,
      "text_encoder_masks": text_encoder_length,

      "text_decoder_targets_positive": text_decoder_length,
      "text_decoder_inputs_positive": text_decoder_length,
      "text_decoder_masks_positive": text_decoder_length,

      "text_decoder_targets_negative": text_decoder_length,
      "text_decoder_inputs_negative": text_decoder_length,
      "text_decoder_masks_negative": text_decoder_length,
    }
    if self.pack:
      model_feature_lengths["text_decoder_segment_ids_positive"] = text_decoder_length
      model_feature_lengths["text_decoder_positions_positive"] = text_decoder_length

      model_feature_lengths["text_decoder_segment_ids_negative"] = text_decoder_length
      model_feature_lengths["text_decoder_positions_negative"] = text_decoder_length

    return model_feature_lengths
