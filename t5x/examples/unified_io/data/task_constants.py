import seqio
import tensorflow as tf

from .data_utils import get_default_vocabulary

DEFAULT_OUTPUT_FEATURES = {
  "image_inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
  "image_targets": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
  "image_input_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_loss_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_inputs":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True),
  "text_targets":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True)
}

FINETUNE_OUTPUT_FEATURES = {
  "image_inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
  "image_targets": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
  "image_input_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_loss_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_decoder_segment_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_decoder_positions": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_inputs":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True),
  "text_targets":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True),
}

FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES = {
  "image_inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
  "image_targets": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
  "image_input_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_target_loss_masks": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_decoder_segment_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_decoder_positions": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "image_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_encoder_pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "text_inputs":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True),
  "text_targets":
    seqio.Feature(
      vocabulary=get_default_vocabulary(), add_eos=True),
}
