import functools
import os

import seqio.preprocessors
import tensorflow as tf
from seqio import TaskRegistry

from .data_utils import get_default_vocabulary
from .metrics import save_predictions
from .preprocessors import text_to_text_preprocessor
from .task_utils import TFDS_DATA_DIR, perplexity
from ...t5.data.mixtures import MixtureRegistry

"""
This file registers datasets related to dialogue. 
"""

@seqio.map_over_dataset
def tokenize(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = {
    'youtube_id': x['youtube_id'],
    'start_sec': x['start_sec'],
    'end_sec': x['end_sec'],
  }
  dialogue = tf.strings.join([x['context'], x['response']], separator='\n')
  pretokenized_dialogue = tf.strings.strip(tf.strings.split(dialogue, '\n'))
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(pretokenized_dialogue), 1])
  out['text_targets_pretokenized'] = dialogue
  out['text_targets'] = tf.concat([voc.encode_tf(pretokenized_dialogue), turn_tokens], axis=1).values

  out['text_inputs_pretokenized'] = tf.strings.strip(x['transcript'])
  out['text_inputs'] = voc.encode_tf(tf.strings.strip(x['transcript']))
  return out


@seqio.map_over_dataset
def yttemporal_tokenize(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = {
    'youtube_id': x['youtube_id'],
    'start_sec': x['start_sec'],
    'end_sec': x['end_sec'],
  }
  target = " "
  out['text_targets_pretokenized'] = target
  out['text_targets'] = voc.encode_tf(target)

  out['text_inputs_pretokenized'] = tf.strings.strip(x['transcript'])
  out['text_inputs'] = voc.encode_tf(tf.strings.strip(x['transcript']))
  return out


# ==== Visual-text dialogue ====
VISUAL_DIALOGUE_DENOISE_TASKS = []

ytdialogue_keys_to_features = {
  'youtube_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'start_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'end_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

yttemporal_keys_to_features = {
  'youtube_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'start_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'end_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
}

DENOISE_OUTPUT_FEATURES = {
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


def add_denoise_dialogue(name, preprocess_fn):
  VISUAL_DIALOGUE_DENOISE_TASKS.append(name)

  if "ytdialogue_denoise" == name:
    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-train.tfrecord*"),
          "validation": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-validation.tfrecord*"),
          "test": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-test.tfrecord*"),
        },
        feature_description=ytdialogue_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'youtube_id',
                                        'start_sec',
                                        'end_sec')),
      ],
      # postprocess_fn=nlp_post_processor,
      output_features=DENOISE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, output_dir='./visual_dialogue_denoise'),
      ],
    )
    TaskRegistry.add(
      f'{name}:metriconly',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-train.tfrecord*"),
          "validation": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-validation.tfrecord*"),
          "test": os.path.join(TFDS_DATA_DIR, 'ytdialogue', "1.0.0", "ytdialogue-test.tfrecord*"),
        },
        feature_description=ytdialogue_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'youtube_id',
                                        'start_sec',
                                        'end_sec')),
      ],
      # postprocess_fn=nlp_post_processor,
      output_features=DENOISE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
      ],
    )
  elif 'yttemporal' in name:
    dataset_name, version, fold = name.split(":")
    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, dataset_name, version, f"yttemporal.tfrecord-{fold}-of-03280"),
        },
        feature_description=yttemporal_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        yttemporal_tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'youtube_id',
                                        'start_sec',
                                        'end_sec')),
      ],
      # postprocess_fn=nlp_post_processor,
      output_features=DENOISE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, output_dir='./visual_dialogue_denoise'),
      ],
    )
  else:
    raise NotImplementedError


@seqio.map_over_dataset
def _preprocess_ytdialogue_denoise(x):
  return {
    'context': x['context'],
    'response': x['response'],
    'transcript': x['transcript'],
    'youtube_id': x['youtube_id'],
    'start_sec': x['start_sec'],
    'end_sec': x['end_sec'],
  }


@seqio.map_over_dataset
def _preprocess_yttemporal(x):
  return {
    'transcript': x['transcript'],
    'youtube_id': x['youtube_id'],
    'start_sec': x['start_sec'],
    'end_sec': x['end_sec'],
  }


add_denoise_dialogue('ytdialogue_denoise', _preprocess_ytdialogue_denoise)
for i in range(3280):
  add_denoise_dialogue(f'yttemporal:1.0.0:{i:05d}', _preprocess_yttemporal)

for i in range(820):
  MixtureRegistry.add(f'yttemporal:1.0.0:chunk~{i:05d}',
                      [f'yttemporal:1.0.0:{j:05d}' for j in range(i * 4, i * 4 + 4)],
                      default_rate=1.0)
