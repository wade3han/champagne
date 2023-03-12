import functools
import os

import seqio.preprocessors
import tensorflow as tf
from seqio import TaskRegistry

from .data_utils import get_default_vocabulary
from .metrics import save_predictions, save_predictions_dialogue
from .postprocessors import visdial_postprocessor, dialogue_postprocessor, dialogue_postprocessor_infer
from .preprocessors import video_to_text_preprocessor, text_to_text_preprocessor
from .task_constants import FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES, FINETUNE_OUTPUT_FEATURES
from .task_utils import store_scores, TFDS_DATA_DIR, perplexity, imagechat_keys_to_features, \
  visdial_keys_to_features, visdial_ndcg_keys_to_features, ytdialogue_multiple_images_keys_to_features, \
  dialogue_metrics
from .utils import tokenize, tokenize_visdial, tokenize_ndcg

"""
This file registers datasets related to dialogue. 
"""


def add_visual_dialogue(name, preprocess_fn):
  if "ytdialogue:1.5.0" == name or 'ytdialogue:1.5.1' == name or 'ytdialogue:1.5.2' == name or 'ytdialogue:1.5.3' == name:
    dataset_name = "ytdialogue"
    version = "1.4.0"
    if name == 'ytdialogue:1.5.0' or name == 'ytdialogue:1.5.2':
      tokenize_fn = tokenize
    elif name == 'ytdialogue:1.5.1':
      tokenize_fn = tokenize_nometa
    elif name == 'ytdialogue:1.5.3':
      tokenize_fn = tokenize_transcript
    else:
      raise NotImplementedError
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )

  elif ("ytdialogue:3.2.0" in name) or ('ytdialogue:3.2.1' in name) or ('ytdialogue:3.2.2' in name) or \
      ('ytdialogue:3.2.3' in name):
    dataset_name, ds_version, num_folds = name.split(":")
    version = "3.1.0"
    valid_version = "1.4.0"
    if ('ytdialogue:3.2.0' in name) or ('ytdialogue:3.2.2' in name):
      tokenize_fn = tokenize
    elif 'ytdialogue:3.2.1' in name:
      tokenize_fn = tokenize_nometa
    elif 'ytdialogue:3.2.3' in name:
      tokenize_fn = tokenize_transcript
    else:
      raise NotImplementedError
    data_num = int(num_folds) // 8 * 45

    TaskRegistry.add(
      f"ytdialogue:{ds_version}:{data_num}k",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )

  elif "ytdialogue:3.3.0" in name:
    dataset_name, ds_version, num_folds = name.split(":")
    version = "3.3.0"
    valid_version = "1.4.0"
    tokenize_fn = tokenize
    data_num = int(num_folds) // 8 * 45

    TaskRegistry.add(
      f"ytdialogue:{ds_version}:{data_num}k",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )

  elif "imagechat:1.2.0:firstturn" == name:
    TaskRegistry.add(
      'imagechat:1.2.0:include_firstturn',
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor,
      metric_fns=[
        perplexity,
        dialogue_metrics,
      ],
    )

    TaskRegistry.add(
      'imagechat:1.2.0:include_firstturn:test',
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor_infer,
      metric_fns=[
        save_predictions_dialogue,
      ],
    )

    TaskRegistry.add(
      'imagechat:1.2.0:firstturn_only:metriconly',
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor,
      metric_fns=[
        perplexity,
        dialogue_metrics,
      ],
    )

    TaskRegistry.add(
      'imagechat:1.2.0:firstturn_only:test',
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      postprocess_fn=dialogue_postprocessor_infer,
      metric_fns=[
        save_predictions_dialogue,
      ],
    )

  elif "visdial:1.2.0" == name:
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='visdial', output_dir='./visdial'),
      ],
    )

  elif 'visdial:1.2.0:ndcg' == name:
    TaskRegistry.add(
      'visdial:1.2.0:ndcg',
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'image_id',
                                        'round_id',
                                        )),
      ],
      postprocess_fn=visdial_postprocessor,
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[store_scores],
    )

  elif 'visdial:1.2.0:val_official' == name or 'visdial:1.2.0:test' == name:
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
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        'image_id',
                                        'round_id',
                                        )),
      ],
      postprocess_fn=visdial_postprocessor,
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[store_scores],
    )

  else:
    raise NotImplementedError


def add_visual_dialogue_no_image(name, preprocess_fn):
  """
  Not using visual information for the ablation.
  """
  if "ytdialogue:1.5.0" == name:
    dataset_name = "ytdialogue"
    version = "1.4.0"
    tokenize_fn = tokenize
    TaskRegistry.add(
      f"{name}:noimage",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )
  elif "ytdialogue:3.2.0" in name:
    dataset_name, ds_version, num_folds = name.split(":")
    version = "3.1.0"
    valid_version = "1.4.0"
    tokenize_fn = tokenize
    TaskRegistry.add(
      f"ytdialogue:{ds_version}:{int(num_folds) // 32 * 180}k:noimage",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          use_multiple_images=True,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )
  else:
    raise NotImplementedError


## ABLATIONS ##
@seqio.map_over_dataset
def tokenize_single_image(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = dict()
  if x['response'].dtype == tf.string:
    out['text_targets_pretokenized'] = x['response']
    out['text_targets'] = voc.encode_tf(tf.strings.strip(x['response']))

  meta = tf.concat([voc.encode_tf('<extra_id_1>'), voc.encode_tf(x['meta'])], axis=0)

  context = tf.strings.split(x['context'], '\n')
  context = tf.strings.strip(context)
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
  out['text_inputs_pretokenized'] = tf.strings.join([x['meta'], x['context']], separator='\t')
  context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
  out['text_inputs'] = tf.concat([meta, context_tokenized.values], axis=0)
  out['image'] = x['image'][2]
  if 'score' in x:
    out['score'] = x['score']
  return out


@seqio.map_over_dataset
def tokenize_nometa(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = dict()
  if x['response'].dtype == tf.string:
    out['text_targets_pretokenized'] = x['response']
    out['text_targets'] = voc.encode_tf(tf.strings.strip(x['response']))

  context = tf.strings.split(x['context'], '\n')
  context = tf.strings.strip(context)
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
  out['text_inputs_pretokenized'] = x['context']
  context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
  out['text_inputs'] = context_tokenized.values
  out['image'] = x['image']
  return out


@seqio.map_over_dataset
def tokenize_transcript(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = dict()
  transcript = x['transcript']
  transcript_tokenized = voc.encode_tf(transcript)
  num_tokens = get_shape_list(transcript_tokenized, expected_rank=1)[0]
  context = tf.slice(transcript_tokenized, [0], [num_tokens // 2])
  response = tf.slice(transcript_tokenized, [num_tokens // 2], [num_tokens // 2])

  out['text_inputs'] = context
  out['text_targets'] = response
  out['image'] = x['image']
  return out


def add_visual_dialogue_single_image(name, preprocess_fn):
  """
  Not using visual information for the ablation.
  """
  if "ytdialogue:1.5.0" == name:
    dataset_name = "ytdialogue"
    version = "1.4.0"
    tokenize_fn = tokenize_single_image
    TaskRegistry.add(
      f"{name}:singleimage",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )
  elif "ytdialogue:3.2.0" in name:
    dataset_name, ds_version, num_folds = name.split(":")
    version = "3.1.0"
    valid_version = "1.4.0"
    tokenize_fn = tokenize_single_image
    TaskRegistry.add(
      f"ytdialogue:{ds_version}:{int(num_folds) // 32 * 180}k:singleimage",
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
        tokenize_fn,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(text_to_text_preprocessor,
                          use_multiple_images=False,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=FINETUNE_OUTPUT_FEATURES,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='ytdialogue', output_dir='./ytdialogue'),
      ],
    )
  else:
    raise NotImplementedError


@seqio.map_over_dataset
def _preprocess_ytdialogue_use_all_multiple_images(x):
  images = tf.stack([x[f'image/{i}'] for i in range(16)])
  num_turns = x['num_turns']
  images = tf.concat([images, images], axis=0)
  images = tf.slice(images, [13 + num_turns], [3])
  return {
    'image': images,
    'context': x['context'],
    'response': x['response'],
    'meta': x['title'],
  }


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None and not tf.executing_eagerly():
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, int):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    raise ValueError(
      "For the tensor `%s`, the actual rank "
      "`%d` (shape = %s) is not equal to the expected rank `%s`" %
      (name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None and not tf.executing_eagerly():
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


@seqio.map_over_dataset
def _preprocess_ytdialogue_use_all_multiple_images_ablation(x):
  images = tf.stack([x[f'image/{i}'] for i in range(16)])
  num_turns = x['num_turns']
  images = tf.concat([images, images], axis=0)
  images = tf.slice(images, [13 + num_turns], [3])
  context = tf.strings.lower(x['context'])
  context = tf.strings.regex_replace(context, '\n', ' ')
  response = tf.strings.lower(x['response'])
  return {
    'image': images,
    'context': context,
    'response': response,
    'meta': x['title'],
  }


@seqio.map_over_dataset
def _preprocess_ytdialogue_use_all_multiple_images_ablation_v2(x):
  images = tf.stack([x[f'image/{i}'] for i in range(16)])
  num_turns = x['num_turns']
  images = tf.concat([images, images], axis=0)
  images = tf.slice(images, [13 + num_turns], [3])
  return {
    'image': images,
    'transcript': x['transcript'],
    'meta': x['title'],
  }


@seqio.map_over_dataset
def _preprocess_multiple_img_visdial(x):
  first_char = tf.strings.substr(x['response'], 0, 1)
  uppercased_first_char = tf.strings.upper(first_char)
  rest_of_string = tf.strings.substr(x['response'], 1, -1)
  processed_strings = uppercased_first_char + rest_of_string + tf.constant('.')

  return {
    'image': tf.expand_dims(x['image'], 0),
    'context': x['context'],
    'response': processed_strings,
    'meta': tf.strings.join([tf.constant('This is a conversation between two people about the picture. Picture:'),
                             x['caption']], separator=' ')
  }


@seqio.map_over_dataset
def _preprocess_multiple_img(x):
  return {
    'image': tf.expand_dims(x['image'], 0),
    'context': x['context'],
    'response': x['response'],
    'meta': x['caption'],
  }


@seqio.map_over_dataset
def _preprocess_visdial_ndcg_multiple_img(x):
  return {
    'image': tf.expand_dims(x['image'], 0),
    'context': x['context'],
    'response': x['response'],
    'image_id': x['image_id'],
    'round_id': x['round_id'],
    'meta': x['caption'],
  }


@seqio.map_over_dataset
def _preprocess_visdial_dense(x):
  return {
    'image': tf.expand_dims(x['image'], 0),
    'context': x['context'],
    'response': x['response'],
    'meta': x['caption'],
    'score': x['score'] - 0.5,
  }


@seqio.map_over_dataset
def _preprocess_visdial_dense_pair(x):
  return {
    'image': tf.expand_dims(x['image'], 0),
    'context': x['context'],
    'positive_response': x['positive_response'],
    'negative_response': x['negative_response'],
    'meta': x['caption'],
  }


@seqio.map_over_dataset
def _preprocess_stats(x):
  images = tf.stack([x[f'image/{i}'] for i in range(16)])
  return {
    'youtube_id': x['youtube_id'],
    'start_sec': x['start_sec'],
    'end_sec': x['end_sec'],

    'context': x['context'],
    'response': x['response'],
    'transcript': x['transcript'],
    'images': images,
  }


add_visual_dialogue('ytdialogue:1.5.0', _preprocess_ytdialogue_use_all_multiple_images)
add_visual_dialogue('imagechat:1.2.0:firstturn', _preprocess_multiple_img)
add_visual_dialogue('visdial:1.2.0', _preprocess_multiple_img_visdial)
add_visual_dialogue('visdial:1.2.0:ndcg', _preprocess_visdial_ndcg_multiple_img)
add_visual_dialogue('visdial:1.2.0:val_official', _preprocess_visdial_ndcg_multiple_img)
add_visual_dialogue('visdial:1.2.0:test', _preprocess_visdial_ndcg_multiple_img)

## START ABLATION ##
add_visual_dialogue_no_image('ytdialogue:1.5.0', _preprocess_ytdialogue_use_all_multiple_images)  # no image
add_visual_dialogue_single_image('ytdialogue:1.5.0', _preprocess_ytdialogue_use_all_multiple_images)  # no image
add_visual_dialogue('ytdialogue:1.5.1', _preprocess_ytdialogue_use_all_multiple_images)  # no meta
add_visual_dialogue('ytdialogue:1.5.2', _preprocess_ytdialogue_use_all_multiple_images_ablation)  # no dialogue
add_visual_dialogue('ytdialogue:1.5.3', _preprocess_ytdialogue_use_all_multiple_images_ablation_v2)  # no dialogue v2
## END ABLATION ##

for i in range(102):
  add_visual_dialogue(f'ytdialogue:3.2.0:{i * 32}', _preprocess_ytdialogue_use_all_multiple_images)
  add_visual_dialogue(f'ytdialogue:3.2.1:{i * 32}', _preprocess_ytdialogue_use_all_multiple_images)  # no meta
  add_visual_dialogue(f'ytdialogue:3.2.2:{i * 32}',
                      _preprocess_ytdialogue_use_all_multiple_images_ablation)  # no dialogue
  add_visual_dialogue(f'ytdialogue:3.2.3:{i * 32}',
                      _preprocess_ytdialogue_use_all_multiple_images_ablation_v2)  # no dialogue v2
  add_visual_dialogue_no_image(f'ytdialogue:3.2.0:{i * 32}', _preprocess_ytdialogue_use_all_multiple_images)  # no image
  add_visual_dialogue_single_image(f'ytdialogue:3.2.0:{i * 32}',
                                   _preprocess_ytdialogue_use_all_multiple_images)  # no image

add_visual_dialogue(f'ytdialogue:3.2.0:3280', _preprocess_ytdialogue_use_all_multiple_images)
add_visual_dialogue(f'ytdialogue:3.2.1:3280', _preprocess_ytdialogue_use_all_multiple_images)
