from typing import Any, Mapping, Optional, Sequence, Tuple

import tensorflow.compat.v2 as tf

from . import autoaugment

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

INPUT_DIM = 224

def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
    image_size: Sequence[int],
    augmentation_settings: Mapping[str, Any],
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns processed and resized images."""

  # Get the image crop.
  if is_training:
    image, im_shape = _decode_and_random_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image, im_shape = _decode_and_center_crop(image_bytes)
  assert image.dtype == tf.uint8

  # Optionally apply RandAugment: https://arxiv.org/abs/1909.13719
  if is_training:
    if augmentation_settings['randaugment'] is not None:
      # Input and output images are dtype uint8.
      image = autoaugment.distort_image_with_randaugment(
          image,
          num_layers=augmentation_settings['randaugment']['num_layers'],
          magnitude=augmentation_settings['randaugment']['magnitude'])

  # Resize and normalize the image crop.
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(
      image, image_size, tf.image.ResizeMethod.BICUBIC)
  image = _normalize_image(image)

  return image, im_shape

def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image

def _decode_and_random_crop(
    image_bytes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Make a random crop of INPUT_DIM."""

  if image_bytes.dtype == tf.dtypes.string:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  else:
    jpeg_shape = tf.shape(image_bytes)
  
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image, im_shape = _distorted_bounding_box_crop(
      image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)

  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image, im_shape = _decode_and_center_crop(image_bytes, jpeg_shape)
  return image, im_shape


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    *,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      jpeg_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = [offset_y, offset_x, target_height, target_width]

  if image_bytes.dtype == tf.dtypes.string:
    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          tf.stack(crop_window),
                                          channels=3)
  else:
    image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)

  im_shape = tf.stack([target_height, target_width])
  return image, im_shape

def _center_crop(image, crop_dim):
  """Center crops an image to a target dimension."""
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = ((image_height - crop_dim) + 1) // 2
  offset_width = ((image_width - crop_dim) + 1) // 2
  return tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_dim, crop_dim)


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    if image_bytes.dtype == tf.dtypes.string:
      jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    else:
      jpeg_shape = tf.shape(image_bytes)

  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  padded_center_crop_size = tf.cast(
      ((INPUT_DIM / (INPUT_DIM + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = [offset_height, offset_width,
                 padded_center_crop_size, padded_center_crop_size]

  if image_bytes.dtype == tf.dtypes.string:
    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          tf.stack(crop_window),
                                          channels=3)
  else:
    image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)

  im_shape = tf.stack([padded_center_crop_size, padded_center_crop_size])
  return image, im_shape


def cutmix_padding(h, w):
  """Returns image mask for CutMix.
  Taken from (https://github.com/google/edward2/blob/master/experimental
  /marginalization_mixup/data_utils.py#L367)
  Args:
    h: image height.
    w: image width.
  """
  r_x = tf.random.uniform([], 0, w, tf.int32)
  r_y = tf.random.uniform([], 0, h, tf.int32)

  # Beta dist in paper, but they used Beta(1,1) which is just uniform.
  image1_proportion = tf.random.uniform([])
  patch_length_ratio = tf.math.sqrt(1 - image1_proportion)
  r_w = tf.cast(patch_length_ratio * tf.cast(w, tf.float32), tf.int32)
  r_h = tf.cast(patch_length_ratio * tf.cast(h, tf.float32), tf.int32)
  bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
  bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
  bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
  bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

  # Create the binary mask.
  pad_left = bbx1
  pad_top = bby1
  pad_right = tf.maximum(w - bbx2, 0)
  pad_bottom = tf.maximum(h - bby2, 0)
  r_h = bby2 - bby1
  r_w = bbx2 - bbx1

  mask = tf.pad(
      tf.ones((r_h, r_w)),
      paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
      mode='CONSTANT',
      constant_values=0)
  mask.set_shape((h, w))
  return mask[..., None]  # Add channel dim.

def my_cutmix(batch):
  """Apply CutMix: https://arxiv.org/abs/1905.04899."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  mask = batch['mask'][:bs]
  images = (mask * batch['images'][:bs] + (1.0 - mask) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = batch['cutmix_ratio'][:bs]
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}

def my_mixup(batch):
  """Apply mixup: https://arxiv.org/abs/1710.09412."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  ratio = batch['mixup_ratio'][:bs, None, None, None]
  images = (ratio * batch['images'][:bs] + (1.0 - ratio) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = ratio[..., 0, 0, 0]  # Unsqueeze
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}


def my_mixup_cutmix(batch):
  """Apply mixup to half the batch, and cutmix to the other."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 4
  mixup_ratio = batch['mixup_ratio'][:bs, None, None, None]
  mixup_images = (mixup_ratio * batch['images'][:bs]
                  + (1.0 - mixup_ratio) * batch['images'][bs:2*bs])
  mixup_labels = batch['labels'][:bs]
  mixup_mix_labels = batch['labels'][bs:2*bs]

  cutmix_mask = batch['mask'][2*bs:3*bs]

  cutmix_images = (cutmix_mask * batch['images'][2*bs:3*bs]
                   + (1.0 - cutmix_mask) * batch['images'][-bs:])
  cutmix_labels = batch['labels'][2*bs:3*bs]
  cutmix_mix_labels = batch['labels'][-bs:]
  cutmix_ratio = batch['cutmix_ratio'][2*bs : 3*bs]

  return {'images': tf.concat([mixup_images, cutmix_images], axis=0),
          'labels': tf.concat([mixup_labels, cutmix_labels], axis=0),
          'mix_labels': tf.concat([mixup_mix_labels, cutmix_mix_labels], 0),
          'ratio': tf.concat([mixup_ratio[..., 0, 0, 0], cutmix_ratio], axis=0)}
