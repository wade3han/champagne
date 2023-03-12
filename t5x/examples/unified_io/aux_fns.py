import functools

import gin
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from t5x import state_utils
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from t5x.examples.unified_io.data.data_utils import load_class_name


@gin.configurable()
def pose_estimation_mask_fn():
  vocab = get_default_vocabulary()
  # return the masks for pose estimation.
  vocab_size = 33152 + 16384
  BIN_START = 32100
  BIN_END = 33100
  label = ['1', '2', '3']
  encoded_label = [vocab.encode(i) for i in label]
  encoded_label = np.array(encoded_label).reshape([-1])
  masks = np.zeros([vocab_size], np.bool_)
  masks[BIN_START:BIN_END] = 1
  masks = jnp.array(masks)
  label_masks = jax.nn.one_hot(encoded_label, vocab_size)
  label_masks = jnp.sum(label_masks, axis=0)
  label_masks = jnp.array(label_masks, jnp.bool_)
  unit_mask = jnp.stack([masks, masks, label_masks], axis=0)
  masks = jnp.tile(unit_mask, [17, 1])
  masks = masks == 0

  return masks


def vqav2_fn():
  class_name = load_class_name('metadata/vqav2/vqa_vocab_all.json')
  vocab_size = 33152 + 16384
  all_name = class_name
  vocab = get_default_vocabulary()
  encoded_name = vocab.encode_tf(all_name)
  encoded_name = encoded_name.to_tensor(default_value=1)
  masks = []
  for i in range(encoded_name.shape[1]):
    masks.append(tf.scatter_nd(encoded_name[:, i:i + 1], tf.ones([tf.shape(encoded_name)[0]], tf.int32), [vocab_size]))
  masks = tf.stack(masks, axis=0)

  masks = jnp.array(masks == 0)
  return masks


def image_tagging_fn():
  class_name = load_class_name('metadata/inauralist/class_name.json')
  vocab_size = 33152 + 16384

  all_name = []
  for name in class_name: all_name += name.split(', ')
  all_name = [i.lower() for i in all_name]
  vocab = get_default_vocabulary()
  encoded_name = vocab.encode_tf(all_name)
  encoded_name = encoded_name.to_tensor(default_value=1)
  masks = []
  for i in range(encoded_name.shape[1]):
    masks.append(tf.scatter_nd(encoded_name[:, i:i + 1], tf.ones([tf.shape(encoded_name)[0]], tf.int32), [vocab_size]))
  masks = tf.stack(masks, axis=0)

  masks = jnp.array(masks == 0)
  return masks


def state_transformation_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'state.*', None),
        # (r'.*/image_projection/kernel/v_col', None),
        # (r'.*/iy_rel_embedding/v_col', None),
        # (r'.*/ix_rel_embedding/v_col', None),
        # (r'.*/it_rel_embedding/v_col', None),
        # (r'.*/ti_rel_embedding/v_col', None),
        # (r'.*discrete_vae.*', None),
        # (r'.*image_embedder.*', None),
        # (r'.*logits_image_dense.*', None),
      ])
  ]

  return fn


def remove_optimizer_state():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'state.*', None),
      ])
  ]
  return fn


def vqgan_restore_scratch_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'state.*', None),
        # (r'state.*', None),
        (r'target.token_embedder.*', None),
        # (r'target.image_embedder.*', None),
        (r'target.encoder.*', None),
        (r'target.decoder.*', None),
      ])
  ]

  return fn


def drop_image_position_embedding_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'target.encoder.image_position_embedding.*', None),
        # (r'target.encoder.latent_embedding.*', None),
        # (r'target.encoder.perceiver_attention.*', None),
      ])
  ]

  return fn


def drop_path_restore_scratch_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        # (r'state.*', None),
      ])
  ]
  return fn


def frozen_champagne_restore_scratch_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'state.*', None),
        (r'target.video_frames_encoder.trainable.*', None),
        (r'target.*encoder.*trainable.*', None),
        # (r'target.token_embedder.*', None),
        # (r'target.encoder.*', None),
        # (r'target.decoder.*', None),
      ])
  ]

  return fn


def frozen_champagne_restore_ablation_fns():
  fn = [
    functools.partial(
      state_utils.apply_assignment_map,
      assignment_map=[
        (r'state.*', None),
        (r'target.video_frames_encoder.*', None),
        (r'target.token_embedder.*', None),
        (r'target.encoder.*', None),
        (r'target.decoder.*', None),
      ])
  ]

  return fn
