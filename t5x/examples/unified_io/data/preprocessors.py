from os import listdir
from os.path import join

import seqio
import json
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import reduce
from absl import logging
import functools
import einops

from .data_utils import *
from .imagenet_utils import _preprocess_image, cutmix_padding, my_cutmix, my_mixup, my_mixup_cutmix
from .prompt_template import *
import gin

AUTOTUNE = tf.data.experimental.AUTOTUNE
rekey = seqio.preprocessors.rekey
tokenize = seqio.preprocessors.tokenize

VOCAB_START = 100
NUM_DETECTION_BIN = 1000
FINETUNE_IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_D = 16
IMAGE_TARGET_SIZE = [256, 256]
IMAGE_TARGET_D = 16
BASE_MASK_PROBS = 0.75
SAMPLE_MASK_PROBS = 0.0
MASKING_UNIT_SIZE = 2

RANDOM_SCALE_MAX = 1.1
RANDOM_SCALE_MIN = 0.9


def get_from_dict(dataDict, mapList):
  """Iterate nested dictionary"""
  return reduce(dict.get, mapList, dataDict)


@seqio.utils.map_over_dataset
def rekey(x, key_map=None):
  """Replace the feature keys according to the mapping in `key_map`.
  For example, if the dataset returns examples of the format:
  {'foo': 'something', 'bar': 'something else'}
  and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
  examples with the format
  {'boo': 'something', 'spar': 'something else'}
  If a mapping is to an empty key or None, set the new key to an empty string.
  Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
  Returns:
      A preprocessed example with the format listed above.
  """
  if key_map:
    return {
      new_key: get_from_dict(x, old_key) if old_key else ''
      for new_key, old_key in key_map.items()
    }
  return x


def build_text_features(text_inputs, target_text, sequence_length):
  segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
  position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

  text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
  text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

  return {
    'text_inputs': text_inputs,
    'text_targets': target_text,
    'text_decoder_segment_ids': segment_ids,
    'text_decoder_positions': position_ids,
    'text_encoder_pos_ids': text_encoder_pos_ids,
  }


def build_image_features(img, img_mask, output_features, sequence_length, image_target=None):
  """Builds images features assuming we do not have a target image to generate,
  from an image that has already been re-sized/padded to be the target input size
  """

  image_input_size = img.shape[:2]
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = tf.constant(
    np.array(image_target_size) / image_target_d, tf.int32)

  if image_target is None:
    image_target = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)
    image_target_masks = tf.zeros(target_padding_size, tf.int32)
    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)
  else:
    # In case the dimension were unknown
    image_target = tf.reshape(image_target, image_target_size + [3])
    image_target_masks = tf.image.resize(
      tf.expand_dims(img_mask, -1),
      target_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)

  image_target = image_target * 2.0 - 1
  image_inputs = normalize_image(img)

  input_padding_size = tf.constant(
    np.array(image_input_size) / image_input_d, tf.int32)

  image_input_masks = tf.image.resize(
    tf.expand_dims(img_mask, 2),
    input_padding_size,
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

  if output_features['image_inputs'].rank == 2:
    image_input_sample_valid = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
    image_input_sample_masked = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

    image_encoder_pos_ids = tf.concat([
      tf.random.shuffle(image_input_sample_valid),
      tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
    image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    image_inputs = einops.rearrange(
      image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=image_target_d, dw=image_target_d)

    image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
    image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
  else:
    image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

  return {
    'image_inputs': image_inputs,
    'image_input_masks': image_input_masks,
    'image_targets': image_target,
    'image_target_masks': image_target_masks,
    'image_target_loss_masks': image_target_masks,
    'image_encoder_pos_ids': image_encoder_pos_ids
  }


def flatten_by_label(ds, target_keys):
  def _flatten(ex):
    labels = ex["label"]
    unique_label, idx = tf.unique(labels)

    def _get_labels(label):
      select = labels == label
      out = dict(ex)
      for k in target_keys:
        if k == 'bbox':
          out[k] = tf.reshape(out[k], [-1, 4])
        out[k] = out[k][select]
      return out

    return tf.data.Dataset.from_tensor_slices(unique_label).map(_get_labels)

  return ds.flat_map(_flatten)


def flatten_parts(ds, parts):
  """
  Flatten `ds` so that the features in `parts` are flattened (means each slice of those
  features becomes and individual example) and the other features in ds are duplicated for each
  flattended example
  """

  def _flatten(ex):
    flat_ds = tf.data.Dataset.from_tensor_slices({k: ex[k] for k in parts})

    def _merge(_flat_ex):
      for k, v in ex.items():
        if k not in parts:
          _flat_ex[k] = v
      return _flat_ex

    return flat_ds.map(_merge)

  return ds.flat_map(_flatten)


@gin.configurable()
def multimodal_prefix_preprocessor(ds, sequence_length, output_features,
                                   image_input_size=IMAGE_INPUT_SIZE,
                                   image_input_d=IMAGE_INPUT_D,
                                   base_mask_probs=BASE_MASK_PROBS,
                                   sample_mask_probs=SAMPLE_MASK_PROBS,
                                   mask_unit_size=MASKING_UNIT_SIZE,
                                   noise_density=0.5,
                                   mean_noise_span_length=3,
                                   random_scale_max=1.1,
                                   random_scale_min=1.0,
                                   random_scale_ratio=0.5,
                                   use_prefix_image=False, use_prefix_text=False, random_caption=False,
                                   decode_jpeg=False, class_name=None, unique_label=False, unique_name=False,
                                   long_text_target=True):
  """Multi-Modal prefix modeling"""
  '''
    preprocessor for image and text pre-training by prefix modeling.
    '''
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  image_input_length = int(image_input_size[0] / image_input_d) * int(image_input_size[1] / image_input_d)
  image_target_length = int(image_target_size[0] / image_target_d) * int(image_target_size[1] / image_target_d)

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  def to_inputs_and_targets(ex):
    image_mask_probs = base_mask_probs + tf.random.uniform([], maxval=sample_mask_probs)
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    do_random_scale=True,
                                                    random_scale_max=random_scale_max,
                                                    random_scale_min=random_scale_min,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=random_scale_ratio,
                                                    resize_method='random')

    # get the mask for image and target. if there is no target, add a fake tenosr.
    img_mask = tf.cast(img_mask, tf.int32)
    image_inputs = img
    image_targets = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_targets = image_targets * 2.0 - 1
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_target_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      target_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if use_prefix_image:
      image_mask_size = tf.cast(target_padding_size / mask_unit_size, tf.int32)
      image_random_mask = tf.random.uniform(shape=image_mask_size, maxval=1) > image_mask_probs
      image_random_mask = tf.expand_dims(tf.expand_dims(image_random_mask, 1), -1)
      image_random_mask = tf.tile(image_random_mask, [1, mask_unit_size, 1, mask_unit_size])
      image_random_mask = tf.reshape(image_random_mask, target_padding_size)

      image_input_random_mask = tf.image.resize(
        tf.cast(image_random_mask[:, :, None], tf.int32),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = image_input_masks * image_input_random_mask

      image_target_loss_masks = tf.cast(image_random_mask == False, tf.int32)[:, :, None]
      image_target_loss_masks = image_target_loss_masks * image_target_masks
      image_target_loss_masks = tf.reshape(image_target_loss_masks, [-1])
    else:
      image_target_loss_masks = tf.zeros([image_target_length], tf.int32)

    image_input_masks = tf.reshape(image_input_masks, [-1])
    image_target_masks = tf.reshape(image_target_masks, [-1])

    # get positions.
    image_input_sample_valid = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
    image_input_sample_masked = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

    image_encoder_pos_ids = tf.concat([
      tf.random.shuffle(image_input_sample_valid),
      tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
    image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    image_inputs = einops.rearrange(
      image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=image_target_d, dw=image_target_d)

    image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
    image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)

    if 'captions' in ex:
      if random_caption:
        rand_int = tf.random.uniform(shape=[], maxval=tf.shape(ex['captions'])[0], dtype=tf.int32)
        sampled_caption = ex['captions'][rand_int]
      else:
        sampled_caption = ex['captions']
    else:
      if 'class_name' in ex:
        class_name = ex['class_name']
      else:
        if unique_label:
          label = tf.cast(ex['label'], tf.int32)
          label = tf.unique(label)[0]
        else:
          label = tf.cast(ex['label'], tf.int32)

        class_name = table.lookup(label)

      if unique_name:
        class_name = tf.strings.split(class_name, sep=', ')
        rand_int = tf.random.uniform(shape=[], maxval=tf.shape(class_name)[0], dtype=tf.int32)
        class_name = class_name[rand_int]

      class_name = tf.strings.lower(class_name)
      sampled_caption = tf.strings.reduce_join(tf.random.shuffle(class_name), separator=', ')

    text_target = tf.strings.lower(sampled_caption)
    vocab = output_features['text_targets'].vocabulary
    text_target = vocab.encode_tf(text_target)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_targets,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_loss_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_targets': text_target,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  prefix_token = output_features['text_targets'].vocabulary.encode_tf('An image of ')
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if use_prefix_text:
    passthrough_feature_keys = [
      'image_inputs', 'image_input_masks',
      'image_targets', 'image_target_masks',
      'image_target_loss_masks',
      'image_encoder_pos_ids', 'text_encoder_pos_ids']

    ds = _denoise(
      ds,
      output_features,
      inputs_fn=_noise_span_to_unique_sentinel,
      targets_fn=_nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
        _random_spans_noise_mask,
        mean_noise_span_length=mean_noise_span_length),
      input_feature_key='text_inputs',
      passthrough_feature_keys=passthrough_feature_keys,
      prefix=prefix_token,
      long_text_target=long_text_target)
  else:
    def to_outputs(ex):
      vocab = output_features['text_targets'].vocabulary
      text_targets = vocab.encode_tf('')
      text_inputs = tf.concat((prefix_token, ex['text_targets']), axis=0)

      return {
        'image_inputs': ex['image_inputs'],
        'image_input_masks': ex['image_input_masks'],
        'image_targets': ex['image_targets'],
        'image_target_masks': ex['image_target_masks'],
        'image_target_loss_masks': ex['image_target_loss_masks'],
        'image_encoder_pos_ids': ex['image_encoder_pos_ids'],
        'text_encoder_pos_ids': ex['text_encoder_pos_ids'],
        'text_targets': text_targets,
        'text_inputs': text_inputs}

    ds = ds.map(to_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


@gin.configurable()
def image_prefix_preprocessor(ds, sequence_length,
                              image_input_size=IMAGE_INPUT_SIZE,
                              image_input_d=IMAGE_INPUT_D,
                              base_mask_probs=BASE_MASK_PROBS,
                              sample_mask_probs=SAMPLE_MASK_PROBS,
                              mask_unit_size=MASKING_UNIT_SIZE,
                              random_scale_max=1.1,
                              random_scale_min=1.0,
                              random_scale_ratio=0.5,
                              decode_jpeg=False):
  """Multi-Modal prefix modeling"""
  '''
  preprocessor for image and text pre-training by prefix modeling.
  '''
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  image_input_length = int(image_input_size[0] / image_input_d) * int(image_input_size[1] / image_input_d)
  image_target_length = int(image_target_size[0] / image_target_d) * int(image_target_size[1] / image_target_d)

  def to_inputs_and_targets(ex):
    image_mask_probs = base_mask_probs + tf.random.uniform([], maxval=sample_mask_probs)
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    do_random_scale=True,
                                                    random_scale_max=random_scale_max,
                                                    random_scale_min=random_scale_min,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=random_scale_ratio,
                                                    resize_method='random')

    # get the mask for image and target. if there is no target, add a fake tenosr.
    img_mask = tf.cast(img_mask, tf.int32)
    image_inputs = img
    image_targets = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_targets = image_targets * 2.0 - 1
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_target_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      target_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_mask_size = tf.cast(target_padding_size / mask_unit_size, tf.int32)
    image_random_mask = tf.random.uniform(shape=image_mask_size, maxval=1) > image_mask_probs
    image_random_mask = tf.expand_dims(tf.expand_dims(image_random_mask, 1), -1)
    image_random_mask = tf.tile(image_random_mask, [1, mask_unit_size, 1, mask_unit_size])
    image_random_mask = tf.reshape(image_random_mask, target_padding_size)

    image_input_random_mask = tf.image.resize(
      tf.cast(image_random_mask[:, :, None], tf.int32),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = image_input_masks * image_input_random_mask

    image_target_loss_masks = tf.cast(image_random_mask == False, tf.int32)[:, :, None]
    image_target_loss_masks = image_target_loss_masks * image_target_masks
    image_target_loss_masks = tf.reshape(image_target_loss_masks, [-1])

    image_input_masks = tf.reshape(image_input_masks, [-1])
    image_target_masks = tf.reshape(image_target_masks, [-1])

    # get positions.
    image_input_sample_valid = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
    image_input_sample_masked = tf.boolean_mask(
      tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

    image_encoder_sample_ids = tf.concat([
      tf.random.shuffle(image_input_sample_valid),
      tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
    image_encoder_sample_ids = tf.reshape(image_encoder_sample_ids, (sequence_length['image_input_samples'],))
    image_encoder_sample_ids = tf.cast(image_encoder_sample_ids, tf.int32)

    image_encoder_pos_ids = tf.concat([
      tf.random.shuffle(image_input_sample_valid),
      tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
    image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    image_inputs = einops.rearrange(
      image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=image_target_d, dw=image_target_d)

    image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
    image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)

    text_inputs = ''
    text_targets = ''
    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_targets,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_loss_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable()
def text_span_corruption(dataset,
                         sequence_length,
                         output_features,
                         mean_noise_span_length=3.0,
                         noise_density=0.15,
                         input_feature_key='text_inputs',
                         merge_examples_to_reduce_padding=True):
  image_target_length = int(IMAGE_TARGET_SIZE[0] / IMAGE_TARGET_D) * int(IMAGE_TARGET_SIZE[1] / IMAGE_TARGET_D)

  """Final pretraining objective used in Raffel et al., 2019."""
  input_length, targets_length = _random_spans_helper(
    extra_tokens_per_span_inputs=1,
    extra_tokens_per_span_targets=1,
    inputs_length=sequence_length[input_feature_key],
    mean_noise_span_length=mean_noise_span_length,
    noise_density=noise_density)

  if sequence_length['text_targets'] < targets_length:
    raise ValueError(
      f'Expected targets length for span corruption ({targets_length}) is '
      f'greater than configured targets length '
      f"({sequence_length['text_targets']})")

  ds = dataset
  ds = _select_random_chunk(
    ds,
    output_features=output_features,
    feature_key='text_targets',
    max_length=65536)

  if merge_examples_to_reduce_padding:
    ds = _reduce_concat_tokens(ds, feature_key='text_targets', batch_size=128)
  ds = _split_tokens(
    ds,
    feature_key='text_targets',
    min_tokens_per_segment=None,
    max_tokens_per_segment=input_length)

  ds = _denoise(
    ds,
    output_features,
    inputs_fn=_noise_span_to_unique_sentinel,
    targets_fn=_nonnoise_span_to_unique_sentinel,
    noise_density=noise_density,
    noise_mask_fn=functools.partial(
      _random_spans_noise_mask,
      mean_noise_span_length=mean_noise_span_length),
    input_feature_key=input_feature_key)

  image_input_size = IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  input_padding_size = int(image_input_size[0] / image_input_d) ** 2
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2

  def to_inputs_and_targets(ex):
    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_inputs = tf.zeros([sequence_length['image_input_samples'], image_input_d ** 2 * 3], tf.float32)
    image_input_masks = tf.zeros([sequence_length['image_input_samples']], tf.int32)
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    image_target_loss_masks = tf.zeros([image_target_length], tf.int32)

    image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_loss_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
      'text_inputs': ex['text_inputs'],
      'text_targets': ex['text_targets'],
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable()
def _random_spans_noise_mask(length,
                             noise_density,
                             seeds,
                             mean_noise_span_length=3.0):
  """Noise mask consisting of random spans of noise tokens.
  The number of noise tokens and the number of noise spans and non-noise spans
  are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
       num_noise_tokens / mean_noise_span_length)
  Spans alternate between non-noise and noise, beginning with non-noise.
  Subject to the above restrictions, all masks are equally likely.
  Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    seeds: an int32 Tensor, shaped (2, 2)
    mean_noise_span_length: a number
  Returns:
    a boolean tensor with shape [length]
  """

  orig_length = length
  # increase length to avoid degeneracy
  length = tf.maximum(length, 2)

  def to_int(x):
    return tf.cast(x, tf.int32)

  def to_float(x):
    return tf.cast(x, tf.float32)

  num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
  num_noise_spans = to_int(
    tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = tf.maximum(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens

  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments, seed):
    """Partition a sequence of items randomly into non-empty segments.
    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      seed: an integer seed
    Returns:
      a Tensor with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = tf.pad(
      seqio.stateless_shuffle(
        to_int(tf.range(num_items - 1) < num_segments - 1),
        seed),
      [[1, 0]])
    segment_id = tf.cumsum(first_in_segment)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length

  noise_span_lengths = _random_segmentation(
    num_noise_tokens, num_noise_spans, seeds[0])
  nonnoise_span_lengths = _random_segmentation(
    num_nonnoise_tokens, num_noise_spans, seeds[1])
  interleaved_span_lengths = tf.reshape(
    tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
    [num_noise_spans * 2])
  span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = tf.math.unsorted_segment_sum(
    tf.ones_like(span_starts), span_starts, length)
  span_num = tf.cumsum(span_start_indicator)
  is_noise = tf.equal(span_num % 2, 1)
  return is_noise[:orig_length]


@gin.configurable()
def sentinel_id(vocabulary, return_value=None):
  """Token ID to use as a sentinel.
  By default, we use the last token in the vocabulary.
  Args:
    vocabulary: a t5.data.vocabularies.Vocabulary
    return_value: an optional integer
  Returns:
    an integer
  """
  if return_value is not None:
    return return_value
  return vocabulary.vocab_size - 1 - NUM_DETECTION_BIN  # for detection.


@gin.configurable()
def _nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  return _noise_span_to_unique_sentinel(
    tokens, tf.logical_not(noise_mask), vocabulary, seeds)


@gin.configurable()
def _noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, seeds):
  """Replace each run of consecutive noise tokens with a different sentinel.
  The idea here is to be able to align the dropped spans in the inputs
  with the markers in the targets.
  We want to generate training examples like
  "We hold X to be Y that" -> "X these truths Y self evident Z"
  Sentinels assigned in decreasing order within the sequence starting at
  vocabulary.size - 1.  That is, we appropriate the last tokens in the
  vocabulary for additional use as sentinels.
  TODO(noam): we may want to try enlarging the vocabulary and leaving room
  for the sentinels instead.  However, this requires enlarging the embedding
  tables in the model, so that is a bigger change.
  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del seeds

  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

  first_noise_tokens = tf.logical_and(
    noise_mask, tf.logical_not(prev_token_is_noise))
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

  sentinel = sentinel_id(vocabulary) + 1 - tf.cumsum(
    tf.cast(first_noise_tokens, tokens.dtype))

  tokens = tf.where(first_noise_tokens, sentinel, tokens)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable
def _reduce_concat_tokens(dataset,
                          feature_key='text_targets',
                          batch_size=128,
                          **unused_kwargs):
  """Token-preprocessor to concatenate multiple unrelated documents.
  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one
  Returns:
    a dataset
  """
  dataset = dataset.map(
    lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


@gin.configurable
def _random_spans_helper(inputs_length=gin.REQUIRED,
                         noise_density=gin.REQUIRED,
                         mean_noise_span_length=gin.REQUIRED,
                         extra_tokens_per_span_inputs=gin.REQUIRED,
                         extra_tokens_per_span_targets=gin.REQUIRED,
                         verbose=False):
  """Training parameters to avoid padding with random_spans_noise_mask.
  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.
  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.
  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.
  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.
  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """

  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
      num_nonnoise_tokens +
      num_noise_spans * extra_tokens_per_span_inputs + 1,
      num_noise_tokens +
      num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length - 1
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
    _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
      'tokens_length=%s inputs_length=%s targets_length=%s '
      'noise_density=%s mean_noise_span_length=%s ',
      tokens_length, inputs_length, targets_length,
      noise_density, mean_noise_span_length)
  return tokens_length, targets_length


def text_prefix_preprocessor(dataset, sequence_length, output_features):
  """Prefix language modeling objective used in Raffel et al. 2019."""

  ds = dataset
  ds = _select_random_chunk(ds, output_features=output_features,
                            feature_key='text_targets', max_length=65536)
  ds = _split_tokens_to_inputs_length(ds, output_features=output_features,
                                      sequence_length=sequence_length)
  ds = _denoise(
    ds,
    output_features,
    inputs_fn=drop_nonnoise_tokens,
    targets_fn=drop_noise_tokens,
    noise_density=0.5,
    noise_mask_fn=random_prefix_noise_mask,
  )

  image_input_size = IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  input_padding_size = int(image_input_size[0] / image_input_d) ** 2
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2

  def to_inputs_and_targets(ex):
    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_inputs = tf.zeros(image_input_size + [3], tf.float32)
    image_input_masks = tf.zeros([input_padding_size], tf.int32)
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'text_inputs': ex['text_inputs'],
      'text_targets': ex['text_targets'],
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable()
def _select_random_chunk(dataset: tf.data.Dataset,
                         output_features: Mapping[str, seqio.Feature],
                         max_length: Optional[int] = None,
                         feature_key: str = 'text_targets',
                         additional_feature_keys: Optional[Sequence[str]] = None,
                         passthrough_feature_keys: Optional[
                           Sequence[str]] = None,
                         sequence_length: Optional[Mapping[str, int]] = None,
                         uniform_random_start: bool = False,
                         min_length: Optional[int] = None,
                         **unused_kwargs) -> tf.data.Dataset:
  """Token-preprocessor to extract one span of at most `max_length` tokens.
  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.
  This is generally followed by split_tokens.
  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key feature_key.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.
  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
        f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  @seqio.map_over_dataset(num_seeds=2)
  def _my_fn(x, seeds):
    """Select a random chunk of tokens.
    Args:
      x: a 1d Tensor
      seeds: an int32 Tensor, shaped (2, 2).
    Returns:
      a 1d Tensor
    """
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_length is not None:
      length = tf.random.stateless_uniform(
        [],
        minval=min_length,
        maxval=max_length,
        dtype=tf.int32,
        seed=seeds[0])
    else:
      length = max_length
    if uniform_random_start:
      start = tf.random.stateless_uniform(
        [],
        minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
        maxval=n_tokens,
        dtype=tf.int32,
        seed=seeds[1])
      end = tf.minimum(start + length, n_tokens)
      start = tf.maximum(start, 0)
    else:
      num_segments = tf.cast(
        tf.math.ceil(
          tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)
        ),
        tf.int32)
      start = length * tf.random.stateless_uniform(
        [],
        maxval=num_segments,
        dtype=tf.int32,
        seed=seeds[1])
      end = tf.minimum(start + length, n_tokens)
    chunk = {feature_key: tokens[start:end]}
    if additional_feature_keys is not None:
      for k in additional_feature_keys:
        with tf.control_dependencies([
          tf.assert_equal(
            tf.shape(tokens)[0],
            tf.shape(x[k])[0],
            message=(f'Additional feature {k} is not the same size as '
                     f'{feature_key} along axis 0 in select_random_chunk().'
                     )
          )
        ]):
          chunk[k] = x[k][start:end]
    if passthrough_feature_keys is not None:
      for k in passthrough_feature_keys:
        chunk[k] = x[k]
    return chunk

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)


@gin.configurable()
def _split_tokens_to_inputs_length(dataset, sequence_length,
                                   output_features, **kwargs):
  max_tokens = sequence_length['text_inputs']
  if output_features['text_inputs'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  return _split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)


@gin.configurable()
def _split_tokens(dataset: tf.data.Dataset,
                  min_tokens_per_segment: Optional[int] = None,
                  max_tokens_per_segment: int = 256,
                  feature_key: str = 'text_targets',
                  additional_feature_keys: Optional[Sequence[str]] = None,
                  passthrough_feature_keys: Optional[Sequence[str]] = None,
                  num_parallel_calls: int = AUTOTUNE,
                  **unused_kwargs) -> tf.data.Dataset:
  """Split examples into multiple examples each.
  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.
  This function is generally preceded by select_random_chunk.
  If min_tokens_per_segment is provided, the segment length is chosen randomly
  per document from a log-uniform distribution.  If min_tokens_per_segment is
  None, then the segment length is max_tokens_per_segment (except for a possibly
  shorter last segment in each document).
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    min_tokens_per_segment: an optional integer
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
    additional_feature_keys: Additional features to split. The same chunk size
      will be used, so they should be the same size as feature_key.
    passthrough_feature_keys: Features to pass through without any splitting.
    num_parallel_calls: num_parallel_calls value to pass to map_over_dataset
  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
        f'split keys {overlap_keys} also included in passthrough keys')

  @seqio.map_over_dataset(num_seeds=1, num_parallel_calls=num_parallel_calls)
  def _split_tokens(x, seed):
    """Split one token sequence into multiple sequences."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # pick a length - log-uniformly distributed
      length = tf.cast(
        tf.exp(
          tf.random.stateless_uniform(
            [],
            minval=math.log(min_tokens_per_segment),
            maxval=math.log(max_tokens_per_segment),
            seed=seed
          )
        ),
        tf.int32)

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
      tf.math.ceil(
        tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32))
      ,
      tf.int32)
    padding = num_segments * length - tf.shape(tokens)[0]

    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      with tf.control_dependencies([
        tf.assert_equal(
          tf.shape(tokens)[0],
          tf.shape(x[k])[0],
          message=(f'Additional feature {k} is not the same size as '
                   f'{feature_key} along axis 0 in split_tokens().')
        )
      ]):
        shape = tf.shape(x[k])[1:]
        padded = tf.pad(
          x[k],
          tf.concat([[[0, padding]],
                     tf.zeros([len(shape), 2], dtype=tf.int32)],
                    axis=0))
        orig_lengths[k] = tf.concat(
          [tf.repeat(length, num_segments - 1), [length - padding]], axis=0)
        outputs[k] = tf.reshape(
          padded, tf.concat([[-1, length], shape], axis=0))
    if passthrough_feature_keys:
      for k in passthrough_feature_keys:
        outputs[k] = tf.tile(
          tf.expand_dims(x[k], axis=0),
          tf.concat([[num_segments], tf.tile([1], [tf.rank(x[k])])], axis=0))
    return outputs, orig_lengths

  def _strip_padding(inputs, orig_lengths):
    output = {}
    for k, v in inputs.items():
      if passthrough_feature_keys and k in passthrough_feature_keys:
        output[k] = v
      else:
        output[k] = v[:orig_lengths[k]]
    return output

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  dataset = _split_tokens(dataset)
  dataset = dataset.unbatch()
  dataset = dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)
  return dataset


@gin.configurable()
def _denoise(dataset,
             output_features,
             noise_density,
             noise_mask_fn,
             inputs_fn,
             targets_fn=None,
             passthrough_feature_keys: Optional[Sequence[str]] = None,
             input_feature_key='text_inputs',
             prefix=None,
             long_text_target=True,
             **unused_kwargs):
  """Gin-configurable token preprocessor for self-supervised denoising tasks.
  This function takes a dataset containing "targets" sequences,
  and turns each sequence into a dictionary containing:
  {
     "inputs": noisy version of the original sequence
     "targets": the full original sequence or missing parts of original sequence
  }
  In particular, for each sequence, we choose a boolean noise_mask identifying
  which tokens in the sequence to corrupt, as defined by the given
  noise_mask_fn.
  Given the sequence and the noise mask, we generate the inputs and targets
  using the given inputs_fn and targets_fn respectively.
  The self-supervised tasks vary along these axes:
    - noise_density: What fraction of the tokens to select as noise
    - noise_mask_fn: What pattern should the noise mask follow
         (iid, regular segments, etc.)
    - inputs_fn: How to apply the noise
         (drop noise tokens, replace with sentinels, etc.)
    - targets_fn: How to represent the output
         (full sequence, only non-noise tokens, etc.)
  Note: Some functionality has been deleted, which we may or may not want to
  restore at a later date.  The code for this functionality can be found in
  the deleted code for this CL.  In particular:
    - mixture of masking and random replacement
    - task labels prepended to the inputs
  Args:
    dataset: A tf.data.Dataset to process.
    output_features: a dict mapping feature name to t5.data.Feature.
    noise_density: a float
    noise_mask_fn: a function from (length, noise_density) -> boolean mask
    inputs_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    targets_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    passthrough_feature_keys: names of additional features to include in output
    input_feature_key: name of feature to use as inputs
  Returns:
    A preprocessed tf.data.Dataset.
  """
  if passthrough_feature_keys and (input_feature_key in passthrough_feature_keys
                                   or 'text_targets' in passthrough_feature_keys):
    raise ValueError(
      f"passthrough keys cannot contain '{input_feature_key}' or 'text_targets'")

  @seqio.map_over_dataset(num_seeds=6)
  def my_fn(features, seeds):
    """Map function."""
    if long_text_target:
      tokens = features['text_targets']
      vocabulary = output_features['text_targets'].vocabulary
      if (input_feature_key in output_features and
          vocabulary != output_features[input_feature_key].vocabulary):
        raise ValueError(
          'denoise creates inputs based on tokenized targets but was applied '
          'to a task that uses different vocabularies for inputs and targets.')
      noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
      inputs = inputs_fn(tokens, noise_mask, vocabulary, seeds=seeds[2:4])
      if targets_fn:
        targets = targets_fn(tokens, noise_mask, vocabulary, seeds=seeds[4:6])
      else:
        targets = tokens

      if prefix is not None:
        inputs = tf.concat((prefix, inputs), axis=0)
    else:
      vocabulary = output_features['text_targets'].vocabulary
      sentinel = tf.constant([sentinel_id(vocabulary)], tf.int32)
      if prefix is not None:
        inputs = tf.concat((prefix, sentinel), axis=0)
      else:
        inputs = sentinel
      targets = tf.concat((sentinel, features['text_targets']), axis=0)

    return {
      input_feature_key: inputs,
      'text_targets': targets,
      **{
        k: features[k]
        for k in features
        if passthrough_feature_keys and k in passthrough_feature_keys
      }
    }

  return my_fn(dataset)


def drop_nonnoise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop non-noise tokens without inserting a sentinel.
  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, noise_mask)


def drop_noise_tokens(tokens, noise_mask, vocabulary, seeds):
  """Drop noise tokens without inserting a sentinel.
  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: an unused vocabulary.Vocabulary
    seeds: an unused int32 Tensor
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  del vocabulary
  del seeds
  return tf.boolean_mask(tokens, tf.logical_not(noise_mask))


def random_prefix_noise_mask(length, noise_density, seeds):
  """First part of the sequence is noise (for prefix_lm).
  The length of the prefix is chosen uniformly between [1, length)
  noise_density must be 0.5
  TODO(noam): figure out some distribution to use if noise_density != 0.5
  Args:
    length: an int32 scalar
    noise_density: a float - must equal 0.5
    seeds: an int32 Tensor, shaped (1, 2)
  Returns:
    a boolean tensor with shape [length]
  """
  if noise_density != 0.5:
    raise NotImplementedError(
      'noise density must equal 0.5 for random_prefix_noise_mask')
  max_input_tokens = length - 1
  min_input_tokens = tf.minimum(max_input_tokens, 1)
  num_input_tokens = tf.random.stateless_uniform(
    [],
    minval=min_input_tokens,
    maxval=max_input_tokens + 1,
    dtype=tf.int32,
    seed=seeds[0])
  return tf.range(length, dtype=tf.int32) < num_input_tokens


def image_caption_preprocessor(ds, sequence_length, output_features, decode_jpeg=False, multiple_caption=False,
                               multiple_targets=False, filter_caption=False, localized_narrative=False,
                               test_split=False):
  '''
  image caption preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  input_padding_size = int(image_input_size[0] / image_input_d) ** 2
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2

  if filter_caption:
    def fileter_fn(ex):
      return ex['caption_num'] > 0

    ds = ds.filter(fileter_fn)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.1,
                                                    random_scale_min=1.0,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_inputs = img
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)

    vocab = output_features['text_targets'].vocabulary
    if localized_narrative:
      text_inputs = tf.strings.join([np.random.choice(Prompt_Image_Localized_Narrative)])
    else:
      text_inputs = tf.strings.join([np.random.choice(Prompt_Image_Captioning)])
    text_inputs = vocab.encode_tf(text_inputs)

    if test_split:
      captions = ''
    else:
      captions = ex['captions']

    if not multiple_caption:
      text_targets = tf.strings.lower(captions)
      text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
      text_targets = tf.strings.regex_replace(text_targets, r"<person>", "person")
      text_targets = vocab.encode_tf(text_targets)
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    else:
      if multiple_targets:
        text_targets = tf.strings.lower(tf.random.shuffle(captions))
        text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
        text_targets = tf.strings.regex_replace(text_targets, r"<person>", "person")
        text_targets = vocab.encode_tf(text_targets)
        text_targets, segment_ids, position_ids = encode_multi_text_targets(
          text_targets, vocab, sequence_length['text_targets'])
        text_inputs = seqio.preprocessors._append_to_innermost_axis(text_inputs, vocab.eos_id)
      else:
        rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(captions)[0], dtype=tf.int32)
        text_targets = tf.strings.lower(captions[rand_idx])
        text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
        text_targets = tf.strings.regex_replace(text_targets, r"<person>", "person")
        text_targets = vocab.encode_tf(text_targets)
        segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
        position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'example_id': ex["image/filename"] if "image/filename" in ex else '0',
      'image_info': this_image_info,
      'all_references': captions,
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def region_caption_preprocessor(ds, sequence_length, output_features, decode_jpeg=False, bbox_format='x1y1x2y2'):
  '''
  region caption preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D

  input_padding_size = int(image_input_size[0] / image_input_d) ** 2
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    boxes = ex['bbox'][None, :]
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    # caption = tf.gather(ex['text'], indices)
    caption = ex['text']
    # rand_int = tf.random.uniform(shape=[], maxval= tf.shape(caption)[0], dtype=tf.int32)
    # boxes = boxes[rand_int:rand_int+1]
    # caption = caption[rand_int]
    region_description, text_labels = convert_bbox_to_sequence(
      boxes,
      None,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
    )

    image_inputs = img
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)

    vocab = output_features['text_targets'].vocabulary
    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Region_Captioning), "{}", region_description)
    text_inputs = vocab.encode_tf(text_inputs)

    text_targets = tf.strings.lower(caption)
    text_targets = vocab.encode_tf(text_targets)
    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'example_id': ex["image/filename"] if "image/filename" in ex else '0',
      'all_references': '',  # ex['captions'],
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  ds = flatten_parts(ds, ["bbox", "text"])
  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def image_tagging_preprocessor(ds, sequence_length, output_features, class_name=None, decode_jpeg=False,
                               unique_label=False, unique_name=False, scene_tagging=False, rand_aug=False):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  is_training = sequence_length.get('is_training', True)
  if rand_aug:
    augmentation_settings = {}
    augmentation_settings['randaugment'] = dict(num_layers=4, magnitude=5)
    augmentation_settings['cutmix'] = False
    augmentation_settings['mixup_alpha'] = None

  def to_inputs_and_targets(ex):
    if rand_aug:
      image_inputs, _ = _preprocess_image(
        ex['image'], is_training, image_input_size, augmentation_settings)
      image_input_masks = tf.ones([int(image_input_size[0] / image_input_d) ** 2], tf.int32)
    else:
      if decode_jpeg:
        img = tf.image.decode_jpeg(ex['image'], channels=3)
      else:
        img = ex['image']

      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                      do_random_scale=is_training,
                                                      random_scale_max=RANDOM_SCALE_MAX,
                                                      random_scale_min=RANDOM_SCALE_MIN,
                                                      shrink_both_sides=True,
                                                      do_flip_if_vertical=False,
                                                      random_scale_ratio=0.5,
                                                      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)
      image_inputs = img
      image_inputs = normalize_image(image_inputs)
      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

      image_input_masks = tf.image.resize(
        tf.expand_dims(img_mask, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    image_targets = tf.zeros(image_target_size + [3], tf.float32)
    image_target_masks = tf.zeros([int(image_target_size[0] / image_target_d) ** 2], tf.int32)

    if 'class_name' in ex:
      class_name = ex['class_name']
    else:
      if unique_label:
        label = tf.cast(ex['label'], tf.int32)
        label = tf.unique(label)
      else:
        label = tf.cast(ex['label'], tf.int32)
      class_name = table.lookup(label)

    class_name = tf.strings.lower(class_name)
    vocab = output_features['text_targets'].vocabulary

    if scene_tagging:
      text_inputs = tf.strings.join([np.random.choice(Prompt_Image_Tagging_Scene)])
    else:
      text_inputs = tf.strings.join([np.random.choice(Prompt_Image_Tagging)])

    text_inputs = vocab.encode_tf(text_inputs)
    all_reference = tf.strings.split(class_name, sep=', ')
    text_targets = tf.random.shuffle(all_reference)
    text_targets = tf.strings.lower(text_targets)
    text_targets = vocab.encode_tf(text_targets)

    text_targets, segment_ids, position_ids = encode_multi_text_targets(
      text_targets, vocab, sequence_length['text_targets'])

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_inputs = seqio.preprocessors._append_to_innermost_axis(text_inputs, vocab.eos_id)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'all_references': all_reference,
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_targets,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def image_generation_preprocessor(ds, sequence_length, output_features, decode_jpeg=False,
                                  random_caption=False, class_name=None, unique_label=False, unique_name=False):
  '''
  image caption preprocessor.
  '''
  image_input_size = IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    image_inputs = tf.zeros(image_input_size + [3], tf.float32)
    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
    image_input_masks = tf.zeros(input_padding_size, tf.int32)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_targets = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_targets = image_targets * 2.0 - 1

    # image target mask is zero (mask all the target)
    target_padding_size = tf.constant(np.array(image_target_size) / image_target_d, tf.int32)
    image_target_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      target_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)

    if 'captions' in ex:
      if random_caption:
        rand_int = tf.random.uniform(shape=[], maxval=tf.shape(ex['captions'])[0], dtype=tf.int32)
        sampled_caption = ex['captions'][rand_int]
      else:
        sampled_caption = ex['captions']
    else:
      if 'class_name' in ex:
        class_name = ex['class_name']
      else:
        if unique_label:
          label = tf.cast(ex['label'], tf.int32)
          label = tf.unique(label)[0]
        else:
          label = tf.cast(ex['label'], tf.int32)

        class_name = table.lookup(label)

      if unique_name:
        class_name = tf.strings.split(class_name, sep=', ')
        rand_int = tf.random.uniform(shape=[], maxval=tf.shape(class_name)[0], dtype=tf.int32)
        class_name = class_name[rand_int]

      class_name = tf.strings.lower(class_name)
      sampled_caption = tf.strings.reduce_join(tf.random.shuffle(class_name), separator=', ')

    sampled_caption = tf.strings.lower(sampled_caption)

    vocab = output_features['text_targets'].vocabulary
    sampled_caption = tf.strings.regex_replace(sampled_caption, r"\\", " ")
    sampled_caption = tf.strings.lower(sampled_caption)

    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Image_Generation), "{}", sampled_caption)
    text_inputs = vocab.encode_tf(text_inputs)
    text_targets = ''
    text_targets = vocab.encode_tf(text_targets)

    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'example_id': ex["image/filename"] if "image/filename" in ex else 0,
      'image_info': image_info,
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_targets,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
      'text_inputs': text_inputs,
      'text_targets': text_targets,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def semantic_segmentation_preprocessor(ds, sequence_length, output_features, class_name=None,
                                       panoptic_image=True, decode_jpeg=False, decode_seg=False, min_ratio=0.01,
                                       has_id=True):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    boxes = tf.reshape(ex['bbox'], [-1, 4])

    if decode_seg:
      segmentation = tf.map_fn(fn=tf.image.decode_png, elems=ex['segmentation'],
                               fn_output_signature=tf.uint8)
      segmentation = segmentation * 255
    else:
      segmentation = ex['segmentation']

    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    masks=segmentation,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    desired_target_size=image_target_size,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / np.prod(image_input_size)
    seg_indices = area > min_ratio
    boxes = boxes[seg_indices]
    labels = labels[seg_indices]
    indices = indices[seg_indices]

    if tf.size(labels) > 0:
      if class_name is not None:
        class_vals_tensor = tf.constant(class_name)
        class_keys_tensor = tf.constant([i for i in range(len(class_name))], dtype=labels.dtype)
        class_table = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(
            class_keys_tensor,
            class_vals_tensor
          ),
          default_value='back_ground'
        )
        text_labels = class_table.lookup(labels[0])
      else:
        assert labels.dtype == tf.dtypes.string
        text_labels = labels[0]

      if panoptic_image:
        ids = tf.gather(ex['id'], indices)
        ids, labels = tf.cond(
          tf.shape(ids)[0] > 0,
          lambda: (ids, labels),
          lambda: (tf.constant([-1], tf.int64), tf.constant([-1], tf.int64))
        )
        segmentation = convert_panoptic_image_to_rgb(masks, ids)
      else:
        masks = tf.gather(masks, indices)
        segmentation, _ = convert_segmentation_to_rgb(masks, labels, class_name)
      valid = True
    else:
      segmentation = tf.zeros(image_target_size + [3, ])
      text_labels = 'none'
      valid = False

    image_target = tf.ensure_shape(segmentation, image_target_size + [3])
    image_target = image_target * 2.0 - 1
    image_inputs = normalize_image(img)

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    if tf.size(labels) > 0:
      image_target_masks = tf.ones(target_padding_size)
    else:
      image_target_masks = tf.zeros(target_padding_size)

    text_labels = tf.strings.regex_replace(text_labels, r"\\", " ")
    text_labels = tf.strings.lower(text_labels)

    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Object_Segmentation), "{}", text_labels)

    text_features = build_text_features(text_inputs, '', sequence_length)

    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)
    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    image_features = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
    }
    features = image_features
    features.update(text_features)
    features['valid'] = valid

    features.update({
      'image_info': image_info,
    })
    return features

  key_dict = ["bbox", "label"]
  if has_id:
    key_dict.append("id")
  if not panoptic_image:
    # Contain per-label binary masks, so we should flatten on these
    key_dict.append("segmentation")
  # Else include the full panoptic images with each match let `to_inputs_and_targets` extract
  # the relevant instances

  ds = flatten_by_label(ds, key_dict)

  def filter_fn(ex):
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ind = area > min_ratio
    return tf.math.reduce_any(ind)

  ds = ds.filter(filter_fn)
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: x["valid"])
  return ds


def shuffle_tensors(*args):
  indices = tf.range(start=0, limit=tf.shape(args[0])[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  return [tf.gather(x, shuffled_indices) for x in args]


def segmentation_based_image_generation_preprocessor(ds, sequence_length, output_features, class_name=None,
                                                     panoptic_image=True, decode_jpeg=False, decode_seg=False,
                                                     min_ratio=0.01, has_id=True):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    boxes = tf.reshape(ex['bbox'], [-1, 4])

    if decode_seg:
      segmentation = tf.map_fn(fn=tf.image.decode_png, elems=ex['segmentation'],
                               fn_output_signature=tf.uint8)
      segmentation = segmentation * 255
    else:
      segmentation = ex['segmentation']

    img, img_mask, this_image_info = resize_and_pad(img, image_target_size,
                                                    boxes=boxes,
                                                    masks=segmentation,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    desired_target_size=image_input_size,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info

    if tf.size(indices) > 0:
      if panoptic_image:
        ids = tf.gather(ex['id'], indices)
        ids, labels = shuffle_tensors(ids, labels)
        segmentation, text_labels = convert_panoptic_image_to_rgb_semantic(masks, ids, labels, class_name)
      else:
        masks = tf.gather(masks, indices)
        masks, labels = shuffle_tensors(masks, labels)
        segmentation, text_labels = convert_segmentation_to_rgb_semantic(masks, labels, class_name)
      valid = True
    else:
      segmentation = tf.zeros(image_input_size + [3, ])
      text_labels = 'none'
      valid = False

    image_inputs = tf.ensure_shape(segmentation, image_input_size + [3])
    image_inputs = normalize_image(image_inputs)
    image_target = img
    image_target = image_target * 2.0 - 1

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)
    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    if tf.size(labels) > 0:
      image_target_masks = tf.ones(target_padding_size)
    else:
      image_target_masks = tf.zeros(target_padding_size)
    text_labels = tf.strings.lower(text_labels)
    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Segmentation_based_Image_Generation), "{}", text_labels)

    text_features = build_text_features(text_inputs, '', sequence_length)

    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)
    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    image_features = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
    }
    features = image_features
    features.update(text_features)
    features['valid'] = valid

    features.update({
      'image_info': image_info,
    })
    return features

  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


def image_inpainting_preprocessor(ds, sequence_length, output_features, class_name=None, decode_jpeg=False,
                                  min_ratio=0.1, bbox_format='y1x1y2x2'):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  def fileter_fn(ex):
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ind = area > min_ratio
    return tf.math.reduce_any(ind)

  ds = ds.filter(fileter_fn)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info

    example = build_image_features(
      img, img_mask, output_features, sequence_length)

    # filter the bbox with min num of pixel.
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid_inds = area > np.prod(image_input_size) * min_ratio
    boxes = boxes[valid_inds]
    labels = labels[valid_inds]

    # random sample one labels.
    if tf.shape(labels)[0] > 0:
      rand_int = tf.random.uniform(shape=[], maxval=tf.shape(labels)[0], dtype=tf.int32)
      label = labels[rand_int]
      box = boxes[rand_int]
      mask = box_mask(tf.cast(box, tf.int32), image_input_size)
      image_input_masks_ori = tf.cast(tf.expand_dims(mask == 0, axis=-1), tf.int32)
      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

      image_input_masks = tf.image.resize(
        tf.cast(tf.expand_dims(img_mask, 2), tf.int32) * image_input_masks_ori,
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)
      region_description, text_labels = convert_bbox_to_sequence(
        tf.expand_dims(box, axis=0),
        tf.expand_dims(label, axis=0),
        class_name,
        num_bin=NUM_DETECTION_BIN,
        image_size=image_input_size[0],
        vocab_start=VOCAB_START,
      )
    else:
      image_input_masks_ori = tf.ones(tf.shape(img)[:2], tf.int32)
      image_input_masks_ori = tf.expand_dims(image_input_masks_ori, axis=-1)
      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
      image_input_masks = tf.image.resize(
        tf.expand_dims(img_mask, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)
      region_description = "none"

    image_inputs = img
    image_inputs = normalize_image(image_inputs)
    image_target = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_target = image_target * 2.0 - 1
    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_target_masks = tf.image.resize(
      tf.cast(image_input_masks_ori == 0, tf.int32),
      target_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)

    region_description = tf.strings.regex_replace(region_description, r"\\", " ")
    region_description = tf.strings.lower(region_description)

    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Image_Inpainting), "{}", region_description)
    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': "",
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def class_specific_detection_preprocessor(ds, sequence_length, output_features,
                                          class_name=None, decode_jpeg=False, bbox_format='y1x1y2x2',
                                          negative_sample_rate=0.0, all_name=None):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  is_training = sequence_length.get('is_training', True)

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
    src_boxes = boxes  # Box before any resizing/padding
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    features = build_image_features(img, img_mask, output_features, sequence_length)
    if tf.shape(boxes)[0] > 0:
      text_targets, text_labels = tf.cond(tf.less(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), negative_sample_rate),
        lambda: generate_random_box(all_name),
        lambda: convert_bbox_to_sequence(
          boxes,
          labels,
          class_name,
          num_bin=NUM_DETECTION_BIN,
          image_size=image_input_size[0],
          vocab_start=VOCAB_START))

      text_labels = tf.strings.regex_replace(text_labels, r"\\", " ")
      text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
      text_targets = tf.strings.lower(text_targets)
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_Object_Detection), "{}",
                                             tf.strings.lower(text_labels))
      features["valid"] = True
    else:
      text_inputs = tf.constant("")
      text_targets = tf.constant("")
      features["valid"] = False

    features.update(build_text_features(text_inputs, text_targets, sequence_length))

    # Needed for evaluation
    features["boxes"] = boxes
    features["image_info"] = image_info
    features["src_boxes"] = src_boxes
    return features

  ds = flatten_by_label(ds, ["bbox", "label"])
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: x["valid"])
  return ds


def detection_preprocessor(ds, sequence_length, output_features, class_name=None, decode_jpeg=False,
                           detect_all_instances=True, bbox_format='y1x1y2x2'):
  if not detect_all_instances:
    raise NotImplementedError("Use `class_specific_detection_preprocessor`")

  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  if class_name is not None:
    class_name_tensor = tf.constant(class_name)
  else:
    class_name_tensor = None

  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    boxes = tf.reshape(ex['bbox'], [-1, 4])

    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    no_target = False
    if not detect_all_instances:
      unique_labels, _ = tf.unique(labels)
      if tf.shape(boxes)[0] > 0:
        rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(unique_labels)[0], dtype=tf.int32)
        select_label = unique_labels[rand_idx]
        ind = labels == select_label
        boxes = boxes[ind]
        labels = labels[ind]
        src_boxes = ex['bbox'][ex['label'] == select_label]
      else:
        no_target = True
        boxes = tf.constant([[0, 0, 0, 0]], tf.float32)
        if labels.dtype == tf.int64:
          labels = tf.constant([-1], tf.int64)
        else:
          labels = tf.constant([""], tf.string)
        src_boxes = boxes
    else:
      if tf.shape(boxes)[0] > 0:
        src_boxes = ex['bbox']
      else:
        no_target = True
        boxes = tf.constant([[0, 0, 0, 0]], tf.float32)
        if labels.dtype == tf.int64:
          labels = tf.constant([-1], tf.int64)
        else:
          labels = tf.constant([""], tf.string)
        src_boxes = boxes

    # convert the boxes and labels to seqs.
    text_targets, text_labels = convert_bbox_to_sequence(
      boxes,
      labels,
      class_name,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
    )

    image_inputs = img
    image_target = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_target = image_target * 2.0 - 1
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_target_masks = tf.zeros(target_padding_size, tf.int32)
    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)

    text_labels = tf.strings.regex_replace(text_labels, r"\\", " ")
    text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")

    if detect_all_instances:
      text_inputs = random.choice(Prompt_Object_Detection_All_Class)
    else:
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_Object_Detection), "{}",
                                             tf.strings.lower(text_labels))

    text_targets = tf.strings.lower(text_targets)
    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if no_target:
      gt_labels = 'none'
    else:
      if class_name_tensor is None:
        gt_labels = labels
      else:
        gt_labels = tf.gather(class_name_tensor, labels)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'example_id': ex["image/filename"] if "image/filename" in ex else '0',
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,

      # Information needed for metrics
      'src_boxes': src_boxes,
      'boxes': boxes,
      'labels': gt_labels,
      'image_info': image_info
    }

  ds = ds.filter(lambda x: tf.size(x['bbox']) > 0)
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: tf.reduce_sum(x["boxes"]) > 0)
  return ds


def box_classification_preprocessor(ds, sequence_length, output_features,
                                    class_name=None, decode_jpeg=False, bbox_format='y1x1y2x2'):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  is_training = sequence_length.get('is_training', True)

  if class_name is not None:
    keys_tensor = tf.constant([i for i in range(len(class_name))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_name
      ),
      default_value='None'
    )

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
    src_boxes = boxes  # Box before any resizing/padding
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    labels=ex['label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    features = build_image_features(img, img_mask, output_features, sequence_length)

    # random sample
    if tf.shape(boxes)[0] > 0:
      rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(boxes)[0], dtype=tf.int32)
      boxes = boxes[rand_idx:rand_idx + 1]
      labels = labels[rand_idx:rand_idx + 1]

      text_boxes, text_labels = convert_bbox_to_sequence(
        boxes,
        labels,
        class_name,
        num_bin=NUM_DETECTION_BIN,
        image_size=image_input_size[0],
        vocab_start=VOCAB_START,
        concat_label_str=False,
      )
      text_targets = tf.strings.regex_replace(text_labels, r"\\", " ")
      text_targets = tf.strings.lower(text_targets)
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_Box_Classification_Scene), "{}",
                                             tf.strings.lower(text_boxes))
      features["valid"] = True
    else:
      text_inputs = tf.constant("")
      text_targets = tf.constant("")
      features["valid"] = False

    features.update(build_text_features(text_inputs, text_targets, sequence_length))

    # Needed for evaluation
    features["boxes"] = boxes
    features["image_info"] = image_info
    features["src_boxes"] = src_boxes
    return features

  ds = flatten_by_label(ds, ["bbox", "label"])
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: x["valid"])
  return ds


def refer_expression_preprocessor(ds, sequence_length, output_features, class_name=None):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  is_training = sequence_length.get('is_training', True)

  def build_features(ex):
    img = ex["image"]
    boxes = tf.expand_dims(ex["bbox"], 0)
    labels = ex["label"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, _, indices = this_image_info

    features = build_image_features(img, img_mask, output_features, sequence_length)

    if tf.shape(boxes)[0] > 0:
      # Each box has multiple labels, we select one at random
      # TODO we could use our multiple-answers trick we use in VQA/captioning here
      c = tf.random.uniform(shape=[], maxval=tf.shape(labels)[0], dtype=tf.int32)
      labels = labels[c:c + 1]

      # convert the boxes and labels text inputs/targets
      text_targets, text_labels = convert_bbox_to_sequence(
        boxes, labels,
        class_name,
        num_bin=NUM_DETECTION_BIN,
        image_size=image_input_size[0],
        vocab_start=VOCAB_START,
      )

      text_labels = tf.strings.regex_replace(text_labels, r"\\", " ")
      text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")

      text_inputs = tf.strings.regex_replace(random.choice(Prompt_Object_Detection), "{}",
                                             tf.strings.lower(text_labels))
      text_targets = tf.strings.lower(text_targets)
      features["valid"] = True
    else:
      text_inputs = tf.constant("")
      text_targets = tf.constant("")
      features["valid"] = False

    features.update(build_text_features(text_inputs, text_targets, sequence_length))

    # Save for use in evaluation
    features["box_size"] = tf.shape(img)[:2]
    features["box"] = boxes
    return features

  ds = flatten_parts(ds, ["bbox", "label"])
  ds = ds.map(build_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: x["valid"])
  return ds


def framenet_flat_preprocessor(ds, output_features, sequence_length):
  is_training = sequence_length.get('is_training', True)
  prompt = 'What is the {} normal of the image ?'

  def _load_images(ex):
    ex["color"] = tf.image.decode_png(ex["color"], channels=3)
    ex["normal"] = tf.image.decode_png(ex["normal"], channels=3)
    return ex

  def _flat_map(ex):
    normal = ex["normal"]
    normal = tf.transpose(normal, (2, 0, 1))
    ids = tf.constant(["x", "y", "z"])
    flat_ds = tf.data.Dataset.from_tensor_slices((normal, ids))

    def _add_to(_normal, _axis):
      out = dict(ex)
      out["full_normal"] = ex["normal"]
      out["normal"] = _normal
      out["axis"] = _axis
      return out

    return flat_ds.map(_add_to)

  def to_inputs_and_targets(ex):
    gt_img = tf.tile(tf.expand_dims(ex["normal"], -1), [1, 1, 3])
    img = tf.image.convert_image_dtype(ex["color"], dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, IMAGE_INPUT_SIZE, masks=tf.image.convert_image_dtype(gt_img, dtype=tf.float32),
      do_random_scale=is_training, random_scale_max=1.2, random_scale_min=0.8,
      shrink_both_sides=True, do_flip_if_vertical=False, random_scale_ratio=0.5,
      desired_target_size=IMAGE_TARGET_SIZE,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, image_target, boxes, _, indices = this_image_info

    features = build_image_features(img, img_mask, output_features, sequence_length, image_target)
    tf_prompt = tf.constant(prompt)
    tf_prompt = tf.strings.regex_replace(tf_prompt, "{}", ex["axis"])

    features.update(build_text_features(
      text_inputs=tf_prompt,
      target_text=tf.constant(""),
      sequence_length=sequence_length
    ))
    features["image_info"] = image_info
    features["gt_image"] = ex["full_normal"]
    features["axis"] = ex["axis"]
    features["example_id"] = ex["example_id"]
    return features

  ds = ds.map(_load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: tf.logical_not(tf.reduce_all(x["normal"] == 128)))
  ds = ds.flat_map(_flat_map)
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


def framenet_preprocessor(ds, output_features, sequence_length):
  image_input_size = IMAGE_INPUT_SIZE
  image_target_size = IMAGE_TARGET_SIZE
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    img = tf.image.decode_png(ex["color"], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    gt_image = tf.image.decode_png(ex["normal"], channels=3)

    target = tf.image.convert_image_dtype(gt_image, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    masks=target,
                                                    do_random_scale=is_training,
                                                    random_scale_max=1.2,
                                                    random_scale_min=0.8,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    desired_target_size=IMAGE_TARGET_SIZE,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, image_target, boxes, _, indices = this_image_info

    features = build_image_features(img, img_mask, output_features, sequence_length, image_target)

    features.update(build_text_features(
      text_inputs=tf.constant(Prompt_Surface_Normals_Estimation[0]),
      target_text=tf.constant(""),
      sequence_length=sequence_length
    ))
    features["image_info"] = image_info
    features["gt_image"] = gt_image
    return features

  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Filter out images that are all invalid
  ds = ds.filter(lambda x: tf.logical_not(tf.reduce_all(x["gt_image"] == 128)))
  return ds


def depth_estimation_preprocessor(ds, output_features, sequence_length,
                                  decode_jpeg=False, max_depth=10, min_depth=0.1,
                                  take=None, depth_png=False):
  image_input_size = IMAGE_INPUT_SIZE
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  image_input_d = IMAGE_INPUT_D
  image_input_length = int(image_input_size[0] / image_input_d) * int(image_input_size[1] / image_input_d)
  is_training = sequence_length.get('is_training', True)

  if take is not None:
    ds = ds.take(take)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      image = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      image = ex['image']

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if depth_png:
      depth = tf.image.decode_png(ex['depth'], channels=1, dtype=tf.uint16)
      depth = tf.cast(depth, tf.float32) / 1000.0

      image = tf.image.resize(
        image,
        tf.shape(depth)[:2],
        method=tf.image.ResizeMethod.BILINEAR)
    else:
      depth = tf.cast(ex['depth'], dtype=tf.float32)

    depth = tf.reshape(depth, tf.shape(image)[:2])
    img, img_mask, this_image_info = resize_and_pad(
      image, image_input_size, masks=tf.expand_dims(depth, -1),
      desired_target_size=image_target_size,
      do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
      random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
      random_scale_ratio=0.5,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_inputs = normalize_image(img)
    image_target = this_image_info[1]

    image_mask_target = tf.image.resize(
      tf.expand_dims(tf.cast(img_mask == 0, tf.float32), -1),
      image_target_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_mask_target = einops.rearrange(
      image_mask_target, '(h dh) (w dw) c -> (h) (w) (dh dw c)',
      dh=image_target_d, dw=image_target_d)

    image_mask_target = tf.math.reduce_sum(image_mask_target, axis=-1)
    image_mask_target = tf.cast(image_mask_target == 0, tf.float32)

    bad_pixel_mask = tf.cast(image_target == 0, tf.float32)
    bad_pixel_mask = einops.rearrange(
      bad_pixel_mask, '(h dh) (w dw) c -> (h) (w) (dh dw c)',
      dh=image_target_d, dw=image_target_d)

    bad_pixel_mask = tf.math.reduce_sum(bad_pixel_mask, axis=-1)
    bad_pixel_mask = tf.cast(bad_pixel_mask == 0, tf.float32)
    image_mask_target = image_mask_target * bad_pixel_mask

    image_target = tf.clip_by_value(image_target, 0, max_depth)
    image_target = image_target / max_depth

    image_target = image_target * 2 - 1
    image_target = tf.tile(image_target, [1, 1, 3])  # convert to color
    image_target = tf.reshape(image_target, image_target_size + [3, ])

    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, -1),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)
    image_target_masks = tf.cast(tf.reshape(image_mask_target, [-1]), tf.int32)

    features = build_text_features(Prompt_Depth_Estimation[0], "", sequence_length)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    features.update({
      "image": ex['image'],
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      # Eval info
      "depth": depth,
      "image_info": this_image_info[0],
    })
    return features

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def rel_predict_preprocessor(ds, output_features, sequence_length, decode_jpeg=False, bbox_format='x1y1x2y2'):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  is_training = sequence_length.get('is_training', True)

  def fileter_fn(ex):
    return tf.size(ex['r_predicate']) > 0

  ds = ds.filter(fileter_fn)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    r_obj_bbox = tf.reshape(ex['r_obj_bbox'], [-1, 4])
    r_sub_bbox = tf.reshape(ex['r_sub_bbox'], [-1, 4])

    if bbox_format == 'x1x2y1y2':
      r_obj_bbox = tf.stack([r_obj_bbox[:, 2], r_obj_bbox[:, 0], r_obj_bbox[:, 3], r_obj_bbox[:, 1]], axis=1)
      r_sub_bbox = tf.stack([r_sub_bbox[:, 2], r_sub_bbox[:, 0], r_sub_bbox[:, 3], r_sub_bbox[:, 1]], axis=1)

    elif bbox_format == 'x1y1x2y2':
      r_obj_bbox = tf.stack([r_obj_bbox[:, 1], r_obj_bbox[:, 0], r_obj_bbox[:, 3], r_obj_bbox[:, 2]], axis=1)
      r_sub_bbox = tf.stack([r_sub_bbox[:, 1], r_sub_bbox[:, 0], r_sub_bbox[:, 3], r_sub_bbox[:, 2]], axis=1)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=r_obj_bbox,
                                                    boxes1=r_sub_bbox,
                                                    labels=ex['r_obj_label'],
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, r_obj_bbox, r_obj_label, indices, r_sub_bbox = this_image_info
    r_sub_label = tf.gather(ex['r_sub_label'], indices)
    r_predicate = tf.gather(ex['r_predicate'], indices)
    r_sub_bbox = tf.gather(r_sub_bbox, indices)

    if tf.shape(r_predicate)[0] > 0:
      rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(r_predicate)[0], dtype=tf.int32)
      r_obj_bbox = r_obj_bbox[rand_idx:rand_idx + 1]
      r_obj_label = r_obj_label[rand_idx:rand_idx + 1]
      r_sub_bbox = r_sub_bbox[rand_idx:rand_idx + 1]
      r_sub_label = r_sub_label[rand_idx:rand_idx + 1]
      r_predicate = r_predicate[rand_idx]
    else:
      r_obj_bbox = tf.zeros([1, 4], tf.float32)
      r_sub_bbox = tf.zeros([1, 4], tf.float32)
      r_obj_label = tf.constant(['none'])
      r_sub_label = tf.constant(['none'])
      r_predicate = tf.constant('none')

    # convert the boxes and labels to seqs.
    obj_targets, obj_labels = convert_bbox_to_sequence(
      r_obj_bbox,
      r_obj_label,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
    )

    sub_targets, sub_labels = convert_bbox_to_sequence(
      r_sub_bbox,
      r_sub_label,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
    )

    image_inputs = img
    image_target = tf.image.resize(
      img,
      image_target_size,
      method=tf.image.ResizeMethod.BICUBIC)

    image_target = image_target * 2.0 - 1
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(
      np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_target_masks = tf.zeros(target_padding_size, tf.int32)
    image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)

    sub_targets = tf.strings.lower(sub_targets)
    obj_targets = tf.strings.lower(obj_targets)

    sub_targets = tf.strings.regex_replace(sub_targets, r"\\", " ")
    obj_targets = tf.strings.regex_replace(obj_targets, r"\\", " ")

    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Relationship_Tagging), "\{1\}", sub_targets)
    text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", obj_targets)

    text_targets = tf.strings.reduce_join([sub_labels, r_predicate, obj_labels], separator=' ')
    text_targets = tf.strings.lower(text_targets)

    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'label': text_targets,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def pose_estimation_preprocessor(ds, sequence_length, output_features):
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  is_training = sequence_length.get('is_training', True)

  def build_features(ex):
    img = ex['image']
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    # [17, 4] the pose coordinates in box format
    keypoint_pos = tf.stack(
      [ex['keypoints'][:, 1], ex['keypoints'][:, 0], ex['keypoints'][:, 1], ex['keypoints'][:, 0]], axis=-1)

    # [1, 4] yxyxy format
    boxes = ex['bbox']
    boxes = tf.expand_dims(tf.stack([boxes[1], boxes[0], boxes[3], boxes[2]], axis=0), 0)

    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    boxes=boxes,
                                                    boxes1=keypoint_pos,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices, keypoint_pos = this_image_info
    features = build_image_features(img, img_mask, output_features, sequence_length)

    if tf.shape(boxes)[0] > 0:
      keypoint_pos = keypoint_pos[:, :2]  # Get keypoints as just points instead of boxes
      keypoint_label = ex['keypoints'][:, 2:3]  # Get the labels as a [17, 1] array
      is_valid = tf.reduce_any(keypoint_label > 0)
    else:
      # Box was removed during image resizing, use dummy values and filter out later
      boxes = tf.constant([[0, 0, 0, 0]], tf.float32)
      keypoint_pos = tf.constant([[0, 0]], tf.float32)
      keypoint_label = tf.constant([[0]], tf.float32)
      is_valid = False

    # convert the boxes and labels to text inputs/outputs
    labels = tf.constant(["person"])
    bbox_str, bbox_labels = convert_bbox_to_sequence(
      boxes,
      labels,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
    )

    pose_str = convert_keypoint_to_sequence(
      keypoint_pos[:, :2],
      keypoint_label,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START)

    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Pose_Estimation), "{}", bbox_str)
    text_parts = build_text_features(text_inputs, pose_str, sequence_length)

    features.update(text_parts)
    features["valid"] = is_valid
    # For evaluation
    features["image"] = ex["image"]
    features["keypoint_pos"] = keypoint_pos
    features["keypoint_label"] = keypoint_label
    return features

  ds = flatten_parts(ds, ["bbox", "keypoints", "id"])  # Per-box examples
  ds = ds.filter(lambda ex: tf.reduce_any(ex['keypoints'][:, 2] > 0))
  ds = ds.map(build_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda ex: ex["valid"])
  return ds


def blended_mvs_preprocessor(ds, sequence_length, output_features):
  def to_inputs_and_targets(ex):
    print(ex.keys())
    raise ValueError()

  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vqa_preprocessor(ds, sequence_length, output_features, answer_mode="multi-target", test=False):
  '''
  VQA preprocessor.
  '''
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    img = ex["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, FINETUNE_IMAGE_INPUT_SIZE,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)
    features = build_image_features(img, img_mask, output_features, sequence_length)

    vocab = output_features['text_targets'].vocabulary
    text_inputs = ex['text_inputs']
    if test:
      text_targets = tf.constant([''] * 10)
    else:
      text_targets = ex['text_targets']
    text_inputs = tf.strings.lower(text_inputs)
    text_targets = tf.strings.lower(text_targets)

    text_inputs = vocab.encode_tf(text_inputs)

    if answer_mode == "multi-target":
      text_targets = tf.random.shuffle(text_targets)
      text_targets = vocab.encode_tf(text_targets)
      text_targets, segment_ids, position_ids = encode_multi_text_targets(
        text_targets, vocab, sequence_length['text_targets'])
      text_inputs = seqio.preprocessors._append_to_innermost_axis(text_inputs, vocab.eos_id)

    elif answer_mode == "random":
      ix = tf.random.uniform(shape=[], maxval=tf.shape(text_targets)[0], dtype=tf.int32)
      text_targets = vocab.encode_tf(text_targets[ix])
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    elif answer_mode == "single-target":
      text_targets = vocab.encode_tf(text_targets)
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    else:
      raise NotImplementedError(answer_mode)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    text_features = {
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }
    features.update(text_features)

    # Save for evaluation purposes
    if test:
      features['all_references'] = tf.constant([''] * 10)
      features['example_id'] = tf.strings.as_string(ex['id'])
      features['image_info'] = this_image_info
    else:
      features['all_references'] = ex['text_targets']

    return features

  if test:
    ds = flatten_parts(ds, ["text_inputs", "id"])  # Get per-question data points
  else:
    ds = flatten_parts(ds, ["text_inputs", "text_targets"])  # Get per-question data points

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vizwiz_vqa_preprocessor(ds, sequence_length, output_features, answer_mode="multi-target", grounding=False):
  '''
  VizWiz VQA preprocessor.
  '''
  is_training = sequence_length.get('is_training', True)
  image_target_size = IMAGE_TARGET_SIZE

  def to_inputs_and_targets(ex):
    img = ex["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    if grounding:
      grounding_mask = ex['mask']
    else:
      grounding_mask = None
    img, img_mask, this_image_info = resize_and_pad(img, FINETUNE_IMAGE_INPUT_SIZE,
                                                    masks=grounding_mask,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=0.5,
                                                    desired_target_size=image_target_size,
                                                    resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info

    if grounding:
      masks = tf.image.convert_image_dtype(masks, dtype=tf.float32)
      masks = tf.tile(masks, [1, 1, 3])

    features = build_image_features(img, img_mask, output_features, sequence_length, image_target=masks)

    vocab = output_features['text_targets'].vocabulary
    text_inputs = ex['text_inputs']
    text_targets = ex['text_targets']
    text_inputs = tf.strings.lower(text_inputs)
    text_targets = tf.strings.lower(text_targets)

    text_inputs = vocab.encode_tf(text_inputs)

    if answer_mode == "multi-target":
      text_targets = tf.random.shuffle(text_targets)
      text_targets = vocab.encode_tf(text_targets)
      text_targets, segment_ids, position_ids = encode_multi_text_targets(
        text_targets, vocab, sequence_length['text_targets'])
      text_inputs = seqio.preprocessors._append_to_innermost_axis(text_inputs, vocab.eos_id)

    elif answer_mode == "random":
      ix = tf.random.uniform(shape=[], maxval=tf.shape(text_targets)[0], dtype=tf.int32)
      text_targets = vocab.encode_tf(text_targets[ix])
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    elif answer_mode == "single-target":
      text_targets = vocab.encode_tf(text_targets)
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    else:
      raise NotImplementedError(answer_mode)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    text_features = {
      'example_id': ex['image/filename'],
      'image_info': image_info,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }
    features.update(text_features)

    # Save for evaluation purposes
    features['all_references'] = ex['text_targets']
    return features

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vcr_preprocessor(ds, sequence_length, output_features,
                     mode='qa', bbox_format='x1y1x2y2',
                     use_multiple_images=False, pass_through=[], eval=False):
  '''
  VCR preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2
  is_training = sequence_length.get('is_training', True)

  if not eval:
    ds = flatten_parts(ds, ["question", "question_bbox_id", "answer_choice",
                            "answer_choice_bbox_id", "rationale_choice", "rationale_choice_bbox_id",
                            "answer_label", "rationale_label"])
  else:
    ds = flatten_parts(ds, ["question", "question_bbox_id", "answer_choice",
                            "answer_choice_bbox_id", "rationale_choice", "rationale_choice_bbox_id",
                            "answer_label", "rationale_label", "annot_ids"])

  def to_inputs_and_targets(ex):
    img = ex["image"]

    boxes = tf.reshape(ex['boxes'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, image_input_size, boxes=boxes, labels=ex['labels'],
      do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX, filter_box=False,
      random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    # convert the boxes and labels to seqs.
    box_str, _ = convert_bbox_to_sequence(
      boxes,
      labels,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
      convert_to_str=False,
      shuffle=False,
    )

    # random sample a question.
    # rand_int = tf.random.uniform(shape=[], maxval= tf.shape(ex['question'])[0], dtype=tf.int32)
    question = ex['question']  # [rand_int]
    question_bbox_id = ex['question_bbox_id']  # [rand_int]
    answer_choice = ex['answer_choice']  # [rand_int]
    answer_choice_bbox_id = ex['answer_choice_bbox_id']  # [rand_int]
    rationale_choice = ex['rationale_choice']  # [rand_int]
    rationale_choice_bbox_id = ex['rationale_choice_bbox_id']  # [rand_int]
    answer_label = ex['answer_label']  # [rand_int]
    rationale_label = ex['rationale_label']  # [rand_int]

    # iteratively replace bbox idx (#0#) to str.
    def replace_id_with_bbox_str(sent, bbox_id, bbox_str):
      i = tf.constant(0)
      num = tf.math.reduce_sum(tf.cast(bbox_id != -1, tf.int32))

      def cond_fn(i, num, sent, bbox_id, bbox_str):
        return i < num

      def body_fn(i, num, sent, bbox_id, bbox_str):
        str_tag = tf.strings.format("#{}#", i)
        str_box = tf.strings.reduce_join(bbox_str[bbox_id[i]], separator=' ')
        sent = tf.strings.regex_replace(sent, str_tag, str_box)
        i += 1
        return i, num, sent, bbox_id, bbox_str

      i, num, sent, bbox_id, bbox_str = tf.while_loop(cond_fn, body_fn, [i, num, sent, bbox_id, bbox_str])
      return sent

    question = replace_id_with_bbox_str(question, question_bbox_id, box_str)

    if mode == 'qa':
      updated_answer_choice = []
      for t in range(4):
        text = replace_id_with_bbox_str(answer_choice[t], answer_choice_bbox_id[t], box_str)
        updated_answer_choice.append(text)
      updated_answer_choice = tf.stack(updated_answer_choice, axis=0)

      text_targets = updated_answer_choice[answer_label]
      text_input_ans = tf.reshape(updated_answer_choice, [-1])
      text_input_ans = tf.strings.reduce_join(text_input_ans, separator=' ')

    elif mode == 'qar':
      text_input_ans = replace_id_with_bbox_str(answer_choice[answer_label], answer_choice_bbox_id[answer_label],
                                                box_str)

      updated_rationale_choice = []
      for t in range(4):
        text = replace_id_with_bbox_str(rationale_choice[t], rationale_choice_bbox_id[t], box_str)
        updated_rationale_choice.append(text)
      updated_rationale_choice = tf.stack(updated_rationale_choice, axis=0)

      text_targets = updated_rationale_choice[rationale_label]

      text_input_rationale = tf.reshape(updated_rationale_choice, [-1])
      text_input_rationale = tf.strings.reduce_join(text_input_rationale, separator=' ')

    image_inputs = img
    image_inputs = normalize_image(image_inputs)
    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    vocab = output_features['text_targets'].vocabulary

    if mode == 'qa':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VCR_QA), "\{1\}", tf.strings.lower(question))
      # text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(text_input_ans))
    elif mode == 'qar':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VCR_QAR), "\{1\}", tf.strings.lower(question))
      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(text_input_ans))
      # text_inputs = tf.strings.regex_replace(text_inputs, "\{3\}", tf.strings.lower(text_input_rationale))

    text_targets = tf.strings.lower(text_targets)
    text_inputs = vocab.encode_tf(text_inputs)
    text_targets = vocab.encode_tf(text_targets)

    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if (not use_multiple_images and output_features['image_inputs'].rank == 2) or \
        (use_multiple_images and output_features['image_inputs'].rank == 3):
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
      if use_multiple_images:
        image_inputs = tf.expand_dims(image_inputs, axis=0)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,

      'all_references': answer_label if mode == 'qa' else rationale_label,
    }
    if eval:
      out['annot_ids'] = ex['annot_ids']
      out['answer_label'] = ex['answer_label']
      out['rationale_label'] = ex['rationale_label']

    for k in pass_through:
      if k in ex:
        out[k] = ex[k]
    return out

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def viscommet_preprocessor(ds, sequence_length, output_features, mode='before', bbox_format='x1y1x2y2',
                           eval=False, use_multiple_images=False):
  '''
  viscommet preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2
  is_training = sequence_length.get('is_training', True)
  option_str = tf.constant(["a: ", "b: ", "c: ", "d: "])

  def to_inputs_and_targets(ex):
    img = ex["image"]

    boxes = tf.reshape(ex['boxes'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, image_input_size, boxes=boxes, labels=ex['names'],
      do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX, filter_box=False,
      random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, boxes, labels, indices = this_image_info
    # convert the boxes and labels to seqs.
    box_str, _ = convert_bbox_to_sequence(
      boxes,
      labels,
      None,
      num_bin=NUM_DETECTION_BIN,
      image_size=image_input_size[0],
      vocab_start=VOCAB_START,
      convert_to_str=False,
      shuffle=False,
    )

    image_inputs = img
    image_inputs = normalize_image(image_inputs)
    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    vocab = output_features['text_targets'].vocabulary

    # iteratively replace bbox idx (#0#) to str.
    def replace_id_with_bbox_str(sent, bbox_id, bbox_str):

      i = tf.constant(0)
      num = tf.math.reduce_sum(tf.cast(bbox_id != -1, tf.int32))

      def cond_fn(i, num, sent, bbox_id, bbox_str):
        return i < num

      def body_fn(i, num, sent, bbox_id, bbox_str):
        str_tag = tf.strings.format("#{}#", i)
        str_box = tf.strings.reduce_join(bbox_str[bbox_id[i] - 1], separator=' ')
        sent = tf.strings.regex_replace(sent, str_tag, str_box)
        i += 1
        return i, num, sent, bbox_id, bbox_str

      i, num, sent, bbox_id, bbox_str = tf.while_loop(cond_fn, body_fn, [i, num, sent, bbox_id, bbox_str])
      return sent

    max_boxes = tf.shape(box_str)[0]
    event_bbox_id = tf.cast(ex['event_bbox_id'], tf.int32)
    event_bbox_id = tf.where(event_bbox_id >= max_boxes, max_boxes, event_bbox_id)
    event = replace_id_with_bbox_str(ex['event'], event_bbox_id, box_str)

    if mode == 'before':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VisComet_Before), "\{1\}", tf.strings.lower(event))
      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(ex['place']))
      text_targets = text_inputs

      if not eval:
        before_bbox_id = tf.cast(ex['before_bbox_id'], tf.int32)
        before_bbox_id = tf.where(before_bbox_id >= max_boxes, max_boxes, before_bbox_id)
        updated_targets = tf.TensorArray(tf.string, size=tf.shape(ex['before'])[0], dynamic_size=True)
        for t in range(tf.shape(ex['before'])[0]):
          text = replace_id_with_bbox_str(ex['before'][t], before_bbox_id[t], box_str)
          updated_targets = updated_targets.write(t, text)
        text_targets = updated_targets.stack()
        text_targets = tf.random.shuffle(text_targets)

    elif mode == 'intent':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VisComet_Intent), "\{1\}", tf.strings.lower(event))
      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(ex['place']))

      if not eval:
        intent_bbox_id = tf.cast(ex['intent_bbox_id'], tf.int32)
        intent_bbox_id = tf.where(intent_bbox_id >= max_boxes, max_boxes, intent_bbox_id)

        updated_targets = tf.TensorArray(tf.string, size=tf.shape(ex['intent'])[0], dynamic_size=True)
        for t in range(tf.shape(ex['intent'])[0]):
          text = replace_id_with_bbox_str(ex['intent'][t], intent_bbox_id[t], box_str)
          updated_targets = updated_targets.write(t, text)
        text_targets = updated_targets.stack()
        text_targets = tf.random.shuffle(text_targets)

    elif mode == 'after':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VisComet_After), "\{1\}", tf.strings.lower(event))
      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(ex['place']))

      if not eval:

        after_bbox_id = tf.cast(ex['after_bbox_id'], tf.int32)
        after_bbox_id = tf.where(after_bbox_id >= max_boxes, max_boxes, after_bbox_id)

        updated_targets = tf.TensorArray(tf.string, size=tf.shape(ex['after'])[0], dynamic_size=True)
        for t in range(tf.shape(ex['after'])[0]):
          text = replace_id_with_bbox_str(ex['after'][t], after_bbox_id[t], box_str)
          updated_targets = updated_targets.write(t, text)
        text_targets = updated_targets.stack()
        text_targets = tf.random.shuffle(text_targets)

    text_inputs = vocab.encode_tf(text_inputs)

    if eval:
      text_targets = ''
      text_targets_encode = vocab.encode_tf(text_targets)
      segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
      position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)
    else:
      text_targets_encode = vocab.encode_tf(text_targets)
      text_targets_encode, segment_ids, position_ids = encode_multi_text_targets(
        text_targets_encode, vocab, sequence_length['text_targets'])

    if output_features['image_inputs'].rank == 2 or (output_features['image_inputs'].rank == 3 and use_multiple_images):
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)
    if use_multiple_images:
      image_inputs = tf.expand_dims(image_inputs, 0)

    return {
      'example_id': ex["id"] + '-###-' + ex['event'],
      'image_info': image_info,
      'all_references': text_targets,
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets_encode,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def imsitu_preprocessor(ds, sequence_length, output_features, mode='verb'):
  '''
  imsitu preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2

  def sample_from_values(values):

    i = tf.constant(0)
    out = ''

    def cond_fn(i, values, out):
      return i < tf.shape(values)[0]

    def body_fn(i, values, out):
      value_list = tf.strings.split(values[i], ', ')
      c = tf.random.uniform(shape=[], maxval=tf.shape(value_list)[0], dtype=tf.int32)
      if i == 0:
        out = tf.strings.reduce_join([out, value_list[c]], separator='')
      else:
        out = tf.strings.reduce_join([out, value_list[c]], separator=',')
      i += 1
      return i, values, out

    i, values, out = tf.while_loop(cond_fn, body_fn, [i, values, out])
    return out

  def to_inputs_and_targets(ex):
    img = ex["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, image_input_size, do_random_scale=True, random_scale_max=1.1,
      random_scale_min=1.05, shrink_both_sides=True, do_flip_if_vertical=False,
      resize_method='random')

    image_info, masks, boxes, labels, indices = this_image_info

    image_inputs = img
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    vocab = output_features['text_targets'].vocabulary

    if mode == 'verb':
      text_inputs = Prompt_Ground_Situation_Recognition_Verb[0]
      text_targets = ex['verb']
    else:
      role_str = tf.strings.reduce_join(ex['role'], separator=', ')
      text_inputs = tf.strings.regex_replace(
        Prompt_Ground_Situation_Recognition_Frame[0], "\{1\}", ex['verb'])

      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", role_str)
      rand_int = tf.random.uniform(shape=[], maxval=tf.shape(ex['values'])[0], dtype=tf.int32)
      value = sample_from_values(ex['values'][rand_int])
      text_targets = tf.stack([ex['role'], tf.strings.split(value, ',')], axis=-1)
      text_targets = tf.strings.reduce_join(text_targets, separator=': ', axis=1)
      text_targets = tf.strings.reduce_join(text_targets, separator='; ')

    text_inputs = vocab.encode_tf(text_inputs)
    text_targets_encode = vocab.encode_tf(text_targets)
    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'example_id': text_targets,
      'image_info': image_info,
      'label': ex['verb'],
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets_encode,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,
      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vis_entailment_preprocessor(ds, sequence_length, output_features):
  '''
  visual entailment preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    img = ex["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
      random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
      random_scale_ratio=0.5,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_inputs = img
    image_inputs = normalize_image(image_inputs)

    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    vocab = output_features['text_targets'].vocabulary

    # t = tf.random.uniform(shape=[], maxval= tf.shape(ex['s1'])[0], dtype=tf.int32)
    s1 = ex['s1']
    s2 = ex['s2']
    label = ex['label']
    text_inputs = tf.strings.regex_replace(random.choice(Prompt_Visual_Entailment), "\{1\}", tf.strings.lower(s1))
    text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(s2))
    text_targets = label
    text_inputs = vocab.encode_tf(text_inputs)
    text_targets = vocab.encode_tf(text_targets)
    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if output_features['image_inputs'].rank == 2:
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    return {
      'label': label,
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

  ds = flatten_parts(ds, ["s1", "s2", "label"])  # Get per-question data points

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def split_line(x):
  x = tf.strings.split(x, sep=b'\t')
  return dict(text_inputs=x[0], text_targets=x[1])


def read_input_target_tsv(filepattern):
  return tf.data.TextLineDataset(filepattern).map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def text_to_text_preprocessor(ds, output_features, sequence_length,
                              pass_through=("example_id",),
                              use_multiple_images=False):
  def _map(ex):
    image_input_size = IMAGE_INPUT_SIZE
    image_input_d = IMAGE_INPUT_D
    image_target_size = IMAGE_TARGET_SIZE
    image_target_d = IMAGE_TARGET_D
    input_padding_size = int(image_input_size[0] / image_input_d) ** 2
    target_padding_size = int(image_target_size[0] / image_target_d) ** 2

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_inputs = tf.zeros(image_input_size + [3], tf.float32)
    image_input_masks = tf.zeros([input_padding_size], tf.int32)
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)

    vocab = output_features['text_targets'].vocabulary
    text_targets = ex['text_targets']
    if not isinstance(text_targets, tf.RaggedTensor):
      text_targets = tf.RaggedTensor.from_tensor(tf.expand_dims(text_targets, 0))
    text_targets, segment_ids, position_ids = encode_multi_text_targets(
      text_targets, vocab, sequence_length['text_targets'])

    if (use_multiple_images and output_features['image_inputs'].rank == 3) or (
        not use_multiple_images and output_features['image_inputs'].rank == 2
    ):
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
      if use_multiple_images:
        image_inputs = tf.expand_dims(image_inputs, axis=0)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_inputs = seqio.preprocessors._append_to_innermost_axis(ex['text_inputs'], vocab.eos_id)
    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

    for k in pass_through:
      if k in ex:
        out[k] = ex[k]
    return out

  return ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def video_to_text_preprocessor(ds, output_features, sequence_length, pass_through=("example_id",),
                               decode_jpeg=False, use_multiple_images=False):
  is_training = sequence_length.get('is_training', True)

  def _map(ex):
    image_input_size = IMAGE_INPUT_SIZE
    image_input_d = IMAGE_INPUT_D
    image_target_size = IMAGE_TARGET_SIZE
    image_target_d = IMAGE_TARGET_D

    vocab = output_features['text_targets'].vocabulary
    text_targets = ex['text_targets']
    if not isinstance(text_targets, tf.RaggedTensor):
      text_targets = tf.RaggedTensor.from_tensor(tf.expand_dims(text_targets, 0))
    text_targets, segment_ids, position_ids = encode_multi_text_targets(
      text_targets, vocab, sequence_length['text_targets'])

    text_inputs = seqio.preprocessors._append_to_innermost_axis(ex['text_inputs'], vocab.eos_id)
    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    def preprocess_image(image):
      if decode_jpeg:
        img = tf.image.decode_jpeg(image, channels=3)
      else:
        img = image

      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      return img

    def process_image(image):
      img = preprocess_image(image)
      img, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      image_inputs = normalize_image(img)
      return image_inputs

    def get_mask(image):
      img = preprocess_image(image)
      _, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

      image_input_masks = tf.image.resize(
        tf.expand_dims(img_mask, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

      # get the mask for image and target. if there is no target, add a fake tenosr.
      target_padding_size = int(image_target_size[0] / image_target_d) ** 2
      image_target = tf.zeros(image_target_size + [3], tf.float32)
      # image target mask is zero (mask all the target)
      image_target_masks = tf.zeros([target_padding_size], tf.int32)
      return image_input_masks, image_target, image_target_masks

    if not use_multiple_images:
      image_inputs = process_image(ex['image'])
      image_input_masks, image_target, image_target_masks = get_mask(ex['image'])

      if output_features['image_inputs'].rank == 2:
        # get positions.
        image_input_sample_valid = tf.boolean_mask(
          tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
        image_input_sample_masked = tf.boolean_mask(
          tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

        image_encoder_pos_ids = tf.concat([
          tf.random.shuffle(image_input_sample_valid),
          tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
        image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
        image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

        image_inputs = einops.rearrange(
          image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
          dh=image_target_d, dw=image_target_d)

        image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
        image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
      else:
        image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
        image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    else:
      image_inputs = tf.map_fn(process_image, ex['image'], fn_output_signature=tf.float32)
      image_input_masks, image_target, image_target_masks = get_mask(ex['image'][0])

      image_inputs = einops.rearrange(
        image_inputs, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d,
      )
      image_encoder_pos_ids = tf.range(image_inputs.shape[1])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

    for k in pass_through:
      if k in ex:
        out[k] = ex[k]
    return out

  return ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _extract_question_and_context(text, vocab):
  split = tf.strings.split(text, tf.constant("question:"))
  context = vocab.encode_tf(tf.strings.strip(split[0]))
  question = tf.strings.join([
    tf.constant(" question: "),
    tf.strings.strip(split[1])
  ])
  question = vocab.encode_tf(question)
  return question, context


def cmumosei_preprocessor(ds, sequence_length, output_features):
  """
  cmumosei preprocessor.
  """
  is_training = sequence_length.get('is_training', True)

  def _map(ex):
    image_input_size = IMAGE_INPUT_SIZE
    image_input_d = IMAGE_INPUT_D
    image_target_size = IMAGE_TARGET_SIZE
    image_target_d = IMAGE_TARGET_D

    vocab = output_features['text_targets'].vocabulary
    text_targets = ex['text_targets']
    if not isinstance(text_targets, tf.RaggedTensor):
      text_targets = tf.RaggedTensor.from_tensor(tf.expand_dims(text_targets, 0))
    text_targets, segment_ids, position_ids = encode_multi_text_targets(
      text_targets, vocab, sequence_length['text_targets'])

    text_inputs = seqio.preprocessors._append_to_innermost_axis(ex['text_inputs'], vocab.eos_id)
    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    def preprocess_image(image):
      img = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return img

    def process_image(image):
      img = preprocess_image(image)
      img, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      image_inputs = normalize_image(img)
      return image_inputs

    def get_mask(image):
      img = preprocess_image(image)
      _, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

      image_input_masks = tf.image.resize(
        tf.expand_dims(img_mask, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

      # get the mask for image and target. if there is no target, add a fake tenosr.
      target_padding_size = int(image_target_size[0] / image_target_d) ** 2
      image_target = tf.zeros(image_target_size + [3], tf.float32)
      # image target mask is zero (mask all the target)
      image_target_masks = tf.zeros([target_padding_size], tf.int32)
      return image_input_masks, image_target, image_target_masks

    image_inputs = tf.map_fn(process_image, ex['image'], fn_output_signature=tf.float32)
    image_input_masks, image_target, image_target_masks = get_mask(ex['image'][0])

    image_inputs = einops.rearrange(
      image_inputs, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
      dh=image_target_d, dw=image_target_d,
    )
    image_encoder_pos_ids = tf.range(image_inputs.shape[1])
    image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,
    }

    return out

  return ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def video_to_text_pair_preprocessor(ds, output_features, sequence_length, pass_through=("example_id",),
                                    decode_jpeg=False, use_multiple_images=False):
  is_training = sequence_length.get('is_training', True)

  def _map(ex):
    image_input_size = IMAGE_INPUT_SIZE
    image_input_d = IMAGE_INPUT_D
    image_target_size = IMAGE_TARGET_SIZE
    image_target_d = IMAGE_TARGET_D

    vocab = output_features['text_targets_positive'].vocabulary

    def get_target(ex, key):
      text_targets = ex[key]
      if not isinstance(text_targets, tf.RaggedTensor):
        text_targets = tf.RaggedTensor.from_tensor(tf.expand_dims(text_targets, 0))
      text_targets, segment_ids, position_ids = encode_multi_text_targets(
        text_targets, vocab, sequence_length['text_targets_positive'])

      return text_targets, segment_ids, position_ids

    text_targets_positive, segment_ids_positive, position_ids_positive = get_target(ex, 'text_targets_positive')
    text_targets_negative, segment_ids_negative, position_ids_negative = get_target(ex, 'text_targets_negative')

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)
    text_inputs = seqio.preprocessors._append_to_innermost_axis(ex['text_inputs'], vocab.eos_id)

    def preprocess_image(image):
      if decode_jpeg:
        img = tf.image.decode_jpeg(image, channels=3)
      else:
        img = image

      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      return img

    def process_image(image):
      img = preprocess_image(image)
      img, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      image_inputs = normalize_image(img)
      return image_inputs

    def get_mask(image):
      img = preprocess_image(image)
      _, img_mask, this_image_info = resize_and_pad(
        img, image_input_size, do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX,
        random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
        random_scale_ratio=0.5,
        resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

      input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)

      image_input_masks = tf.image.resize(
        tf.expand_dims(img_mask, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

      # get the mask for image and target. if there is no target, add a fake tenosr.
      target_padding_size = int(image_target_size[0] / image_target_d) ** 2
      image_target = tf.zeros(image_target_size + [3], tf.float32)
      # image target mask is zero (mask all the target)
      image_target_masks = tf.zeros([target_padding_size], tf.int32)
      return image_input_masks, image_target, image_target_masks

    if not use_multiple_images:
      image_inputs = process_image(ex['image'])
      image_input_masks, image_target, image_target_masks = get_mask(ex['image'])

      if output_features['image_inputs'].rank == 2:
        # get positions.
        image_input_sample_valid = tf.boolean_mask(
          tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
        image_input_sample_masked = tf.boolean_mask(
          tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

        image_encoder_pos_ids = tf.concat([
          tf.random.shuffle(image_input_sample_valid),
          tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
        image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
        image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

        image_inputs = einops.rearrange(
          image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
          dh=image_target_d, dw=image_target_d)

        image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
        image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
      else:
        image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
        image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    else:
      image_inputs = tf.map_fn(process_image, ex['image'], fn_output_signature=tf.float32)
      image_input_masks, image_target, image_target_masks = get_mask(ex['image'][0])

      image_inputs = einops.rearrange(
        image_inputs, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d,
      )
      image_encoder_pos_ids = tf.range(image_inputs.shape[1])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,

      'text_targets_positive': text_targets_positive,
      'text_decoder_segment_ids_positive': segment_ids_positive,
      'text_decoder_positions_positive': position_ids_positive,

      'text_targets_negative': text_targets_negative,
      'text_decoder_segment_ids_negative': segment_ids_negative,
      'text_decoder_positions_negative': position_ids_negative,
    }

    for k in pass_through:
      if k in ex:
        out[k] = ex[k]
    return out

  return ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def vcr_v2_preprocessor(ds, sequence_length, output_features,
                        mode='qa', bbox_format='x1y1x2y2',
                        use_multiple_images=False, pass_through=[], eval=False):
  '''
  VCR preprocessor.
  '''
  image_input_size = FINETUNE_IMAGE_INPUT_SIZE
  image_input_d = IMAGE_INPUT_D
  image_target_size = IMAGE_TARGET_SIZE
  image_target_d = IMAGE_TARGET_D
  target_padding_size = int(image_target_size[0] / image_target_d) ** 2
  is_training = sequence_length.get('is_training', True)

  if not eval:
    ds = flatten_parts(ds, ["question", "answer_choice",
                            "rationale_choice", "answer_label", "rationale_label"])
  else:
    ds = flatten_parts(ds, ["question", "answer_choice",
                            "rationale_choice", "answer_label", "rationale_label", "annot_ids"])

  def to_inputs_and_targets(ex):
    img = ex["image"]

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(
      img, image_input_size, boxes=None, labels=ex['labels'],
      do_random_scale=is_training, random_scale_max=RANDOM_SCALE_MAX, filter_box=False,
      random_scale_min=RANDOM_SCALE_MIN, shrink_both_sides=True, do_flip_if_vertical=False,
      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    image_info, masks, _, labels, indices = this_image_info

    # random sample a question.
    # rand_int = tf.random.uniform(shape=[], maxval= tf.shape(ex['question'])[0], dtype=tf.int32)
    question = ex['question']  # [rand_int]
    answer_choice = ex['answer_choice']  # [rand_int]
    rationale_choice = ex['rationale_choice']  # [rand_int]
    answer_label = ex['answer_label']  # [rand_int]
    rationale_label = ex['rationale_label']  # [rand_int]

    if mode == 'qa':
      updated_answer_choice = []
      for t in range(4):
        updated_answer_choice.append(answer_choice[t])
      updated_answer_choice = tf.stack(updated_answer_choice, axis=0)

      text_targets = updated_answer_choice[answer_label]
      text_input_ans = tf.reshape(updated_answer_choice, [-1])
      text_input_ans = tf.strings.reduce_join(text_input_ans, separator=' ')

    elif mode == 'qar':
      text_input_ans = answer_choice[answer_label]

      updated_rationale_choice = []
      for t in range(4):
        text = rationale_choice[t]
        updated_rationale_choice.append(text)
      updated_rationale_choice = tf.stack(updated_rationale_choice, axis=0)

      text_targets = updated_rationale_choice[rationale_label]

    image_inputs = img
    image_inputs = normalize_image(image_inputs)
    input_padding_size = tf.constant(np.array(image_input_size) / image_input_d, tf.int32)
    image_input_masks = tf.image.resize(
      tf.expand_dims(img_mask, 2),
      input_padding_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # get the mask for image and target. if there is no target, add a fake tenosr.
    image_target = tf.zeros(image_target_size + [3], tf.float32)
    # image target mask is zero (mask all the target)
    image_target_masks = tf.zeros([target_padding_size], tf.int32)
    vocab = output_features['text_targets'].vocabulary

    if mode == 'qa':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VCR_QA), "\{1\}", tf.strings.lower(question))
    elif mode == 'qar':
      text_inputs = tf.strings.regex_replace(random.choice(Prompt_VCR_QAR), "\{1\}", tf.strings.lower(question))
      text_inputs = tf.strings.regex_replace(text_inputs, "\{2\}", tf.strings.lower(text_input_ans))

    text_targets = tf.strings.lower(text_targets)
    text_inputs = vocab.encode_tf(text_inputs)
    text_targets = vocab.encode_tf(text_targets)

    segment_ids = tf.ones((sequence_length['text_targets'],), dtype=tf.int32)
    position_ids = tf.range(sequence_length['text_targets'], dtype=tf.int32)

    if (not use_multiple_images and output_features['image_inputs'].rank == 2) or \
        (use_multiple_images and output_features['image_inputs'].rank == 3):
      # get positions.
      image_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks)
      image_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(image_input_masks)[0]), image_input_masks == 0)

      image_encoder_pos_ids = tf.concat([
        tf.random.shuffle(image_input_sample_valid),
        tf.random.shuffle(image_input_sample_masked)], axis=0)[:sequence_length['image_input_samples']]
      image_encoder_pos_ids = tf.reshape(image_encoder_pos_ids, (sequence_length['image_input_samples'],))
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

      image_inputs = einops.rearrange(
        image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
        dh=image_target_d, dw=image_target_d)

      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
      if use_multiple_images:
        image_inputs = tf.expand_dims(image_inputs, axis=0)
    else:
      image_encoder_pos_ids = tf.range(sequence_length['image_input_samples'])
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    text_encoder_pos_ids = tf.range(sequence_length['text_inputs'])
    text_encoder_pos_ids = tf.cast(text_encoder_pos_ids, tf.int32)

    out = {
      'image_inputs': image_inputs,
      'image_input_masks': image_input_masks,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'image_target_loss_masks': image_target_masks,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
      'text_decoder_segment_ids': segment_ids,
      'text_decoder_positions': position_ids,

      'image_encoder_pos_ids': image_encoder_pos_ids,
      'text_encoder_pos_ids': text_encoder_pos_ids,

      'all_references': answer_label if mode == 'qa' else rationale_label,
    }
    if eval:
      out['annot_ids'] = ex['annot_ids']
      out['answer_label'] = ex['answer_label']
      out['rationale_label'] = ex['rationale_label']

    for k in pass_through:
      if k in ex:
        out[k] = ex[k]
    return out

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
