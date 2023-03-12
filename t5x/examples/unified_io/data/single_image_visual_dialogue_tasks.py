import functools
import os

import seqio.preprocessors
import tensorflow as tf
from seqio import TaskRegistry

from .metrics import save_predictions, save_predictions_dialogue
from .postprocessors import visdial_postprocessor, dialogue_postprocessor, dialogue_postprocessor_infer
from .preprocessors import video_to_text_preprocessor, text_to_text_preprocessor
from .task_constants import FINETUNE_OUTPUT_FEATURES
from .task_utils import perplexity, store_scores, TFDS_DATA_DIR, imagechat_keys_to_features, \
  visdial_keys_to_features, visdial_ndcg_keys_to_features, ytdialogue_keys_to_features, \
  ytdialogue_multiple_images_keys_to_features, dialogue_metrics
from .utils import tokenize, tokenize_visdial, tokenize_ndcg

"""
This file registers datasets related to dialogue. 
"""


def add_visual_dialogue(name, preprocess_fn):
  if "ytdialogue:1.4.0" == name:
    dataset_name, version = name.split(":")
    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": os.path.join(TFDS_DATA_DIR, dataset_name, version, f"{dataset_name}-train.tfrecord*"),
          "validation": os.path.join(TFDS_DATA_DIR, dataset_name, version, f"{dataset_name}-validation.tfrecord*"),
          "test": os.path.join(TFDS_DATA_DIR, dataset_name, version, f"{dataset_name}-test.tfrecord*"),
        },
        feature_description=ytdialogue_multiple_images_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized')),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )

  elif "ytdialogue:3.1.0" in name:
    dataset_name, version, num_folds = name.split(":")
    valid_version = "1.4.0"

    TaskRegistry.add(
      f"ytdialogue:3.1.0:{int(num_folds) // 32 * 180}k",
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": [os.path.join(TFDS_DATA_DIR, dataset_name, version,
                                 f"{dataset_name}-train.tfrecord-{fold:05d}-of-03280") for fold in
                    range(int(num_folds))],
          "validation": os.path.join(TFDS_DATA_DIR, 'ytdialogue', valid_version, f"ytdialogue-validation.tfrecord*"),
        },
        feature_description=ytdialogue_multiple_images_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized')),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )

  elif "imagechat:1.1.0:firstturn" == name:
    TaskRegistry.add(
      'imagechat:1.1.0:include_firstturn',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": os.path.join(TFDS_DATA_DIR, "imagechat_include_firstturn", "1.0.0", f"imagechat-train.tfrecord*"),
          "validation": os.path.join(TFDS_DATA_DIR, "imagechat_include_firstturn", "1.0.0",
                                     f"imagechat-valid.tfrecord*"),
          "test": os.path.join(TFDS_DATA_DIR, "imagechat_include_firstturn", "1.0.0", f"imagechat-test.tfrecord*"),
        },
        feature_description=imagechat_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor,
      metric_fns=[
        perplexity,
        dialogue_metrics,
      ],
    )

    TaskRegistry.add(
      'imagechat:1.1.0:include_firstturn:test',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, "imagechat_include_firstturn", "1.0.0", f"imagechat-test.tfrecord*"),
        },
        feature_description=imagechat_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor_infer,
      metric_fns=[
        save_predictions_dialogue,
      ]
    )

    TaskRegistry.add(
      'imagechat:1.1.0:firstturn_only:metriconly',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, "imagechat_firstturn_only", "1.0.0",
                               f"imagechat-test.tfrecord*"),
        },
        feature_description=imagechat_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor,
      metric_fns=[
        perplexity,
        dialogue_metrics,
      ],
    )

    TaskRegistry.add(
      'imagechat:1.1.0:firstturn_only:test',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, "imagechat_firstturn_only", "1.0.0",
                               f"imagechat-test.tfrecord*"),
        },
        feature_description=imagechat_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor_infer,
      metric_fns=[
        save_predictions_dialogue,
      ],
    )

  elif "visdial:1.1.0" == name:
    ds_name = "visdial"
    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "train": os.path.join(TFDS_DATA_DIR, ds_name, "1.1.0", f"{ds_name}-train.tfrecord*"),
          "validation": os.path.join(TFDS_DATA_DIR, ds_name, "1.1.0", f"{ds_name}-val.tfrecord*"),
          "test": os.path.join(TFDS_DATA_DIR, ds_name, "1.1.0", f"{ds_name}-val.tfrecord*"),
        },
        feature_description=visdial_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize_visdial,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized')),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='visdial', output_dir='./visdial'),
      ],
    )

  elif 'visdial:1.1.0:ndcg' == name:
    TaskRegistry.add(
      'visdial:1.1.0:ndcg',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, "visdial", "1.1.0", "visdial-val-ndcg.tfrecord*"),
        },
        feature_description=visdial_ndcg_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize_ndcg,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'image_id',
                                        'round_id')),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[store_scores],
    )

  elif 'visdial:1.1.0:val_official' == name or 'visdial:1.1.0:test' == name:
    split = name.split(':')[-1]
    if split == 'val_official':
      split = 'val-official'
    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          "test": os.path.join(TFDS_DATA_DIR, "visdial", "1.1.0", f"visdial-{split}.tfrecord*"),
        },
        feature_description=visdial_ndcg_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize_ndcg,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'image_id',
                                        'round_id')),
      ],
      postprocess_fn=visdial_postprocessor,
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[store_scores],
    )

  else:
    raise NotImplementedError


@seqio.map_over_dataset
def _preprocess_ytdialogue(x):
  return {
    'image': x['image'],
    'context': x['context'],
    'response': x['response'],
    'meta': x['title'],
  }


@seqio.map_over_dataset
def _preprocess_ytdialogue_pick_from_multiple_images(x):
  num_turns = x['num_turns']
  images = tf.stack([x[f'image/{i}'] for i in range(16)])
  image = images[num_turns - 2]
  return {
    'image': image,
    'context': x['context'],
    'response': x['response'],
    'meta': x['title'],
  }


@seqio.map_over_dataset
def _preprocess_single_img(x):
  return {
    'image': x['image'],
    'context': x['context'],
    'response': x['response'],
    'meta': x['caption'],
  }


@seqio.map_over_dataset
def _preprocess_single_img_visdial(x):
  first_char = tf.strings.substr(x['response'], 0, 1)
  uppercased_first_char = tf.strings.upper(first_char)
  rest_of_string = tf.strings.substr(x['response'], 1, -1)
  processed_strings = uppercased_first_char + rest_of_string + tf.constant('.')

  return {
    'image': x['image'],
    'context': x['context'],
    'response': processed_strings,
    'meta': tf.strings.join([tf.constant('This is a conversation between two people about the picture. Picture:'),
                             x['caption']], separator=' ')
  }


@seqio.map_over_dataset
def _preprocess_visdial_ndcg(x):
  return {
    'image': x['image'],
    'context': x['context'],
    'response': x['response'],
    'image_id': x['image_id'],
    'round_id': x['round_id'],
    'meta': x['caption'],
  }


add_visual_dialogue('ytdialogue:1.4.0', _preprocess_ytdialogue_pick_from_multiple_images)
add_visual_dialogue('imagechat:1.1.0:firstturn', _preprocess_single_img)
add_visual_dialogue('visdial:1.1.0', _preprocess_single_img_visdial)
add_visual_dialogue('visdial:1.1.0:ndcg', _preprocess_visdial_ndcg)
add_visual_dialogue('visdial:1.1.0:val_official', _preprocess_visdial_ndcg)
add_visual_dialogue('visdial:1.1.0:test', _preprocess_visdial_ndcg)

for i in range(102):
  add_visual_dialogue(f'ytdialogue:3.1.0:{i * 32}', _preprocess_ytdialogue_pick_from_multiple_images)
