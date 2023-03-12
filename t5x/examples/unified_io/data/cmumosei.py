import functools
import os
from typing import Sequence

import numpy as np
import seqio.preprocessors
from seqio import TaskRegistry
from seqio.metrics import Scalar

from .metrics import save_predictions
from .preprocessors import video_to_text_preprocessor
from .task_constants import *
from .task_utils import TFDS_DATA_DIR, perplexity, cmumosei_keys_to_features, cmumosei_label_keys_to_features
from ..evaluator import UnifiedIOOutput

"""
This file registers datasets related to dialogue. 
"""


def mosei_exact_match(targets: Sequence, predictions: Sequence[UnifiedIOOutput],
                      aux_values, print_examples=True):
  # Use F1, WA, Accuarcy for MOSEI
  if isinstance(targets[0], dict):
    targets = [x["text_target"] for x in targets]

  matches = np.array([target.lower() == pred.text.lower() for target, pred in zip(targets, predictions)])
  labels = np.array([1 if "yes" in target.lower() else 0 for target in targets])
  num_positive = np.sum(labels == 1, dtype=np.float)
  num_true_positive = np.sum((labels == 1) & (matches == 1), dtype=np.float)
  num_false_positive = np.sum((labels == 0) & (matches == 0), dtype=np.float)
  num_negative = np.sum(labels == 0, dtype=np.float)
  num_true_negative = np.sum((labels == 0) & (matches == 1), dtype=np.float)
  num_false_negative = np.sum((labels == 1) & (matches == 0), dtype=np.float)
  print(f"Num pos: {num_true_positive} / {num_positive}, Num neg: {num_true_negative} / {num_negative}, ")
  weighted_average = (num_true_positive / num_positive + num_true_negative / num_negative) * 0.5
  f1_score = (2 * num_true_positive) / (2 * num_true_positive + num_false_positive + num_false_negative)
  neg_f1_score = (2 * num_true_negative) / (2 * num_true_negative + num_false_positive + num_false_negative)
  weighted_f1_score = (f1_score * num_positive + neg_f1_score * num_negative) / (num_positive + num_negative)

  if print_examples:
    ixs = np.random.choice(len(targets), min(20, len(targets)), replace=False)
    examples = [f"pred={predictions[i].text.lower()} gt={targets[i]}" for i in ixs]
    for ex in examples:
      print(ex)
  return {
    "mean_average": Scalar(np.mean(matches)),
    "weighted_average": Scalar(weighted_average),
    "f1_score": Scalar(f1_score),
    "weighted_f1_score": Scalar(weighted_f1_score),
  }


@seqio.map_over_dataset
def tokenize(x):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = dict()
  if x['response'].dtype == tf.string:
    out['text_targets_pretokenized'] = x['response']
    out['text_targets'] = voc.encode_tf(tf.strings.strip(x['response']))

  out['text_inputs_pretokenized'] = x['context']
  out['text_inputs'] = voc.encode_tf(x['context'])
  out['image'] = x['image']
  return out


def add_cmu_mosei(name, preprocess_fn):
  if "cmumosei" in name:
    version = name.split(':')[-1]
    if version == '1.0.0':
      use_multiple_images = True
      output_features = FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES
    elif version == '1.0.1':
      use_multiple_images = False
      output_features = FINETUNE_OUTPUT_FEATURES
    else:
      raise NotImplementedError

    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          'train': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                                'cmumosei-train.tfrecord*'),
          'validation': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                                     'cmumosei-val.tfrecord*'),
          'test': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                               'cmumosei-test.tfrecord*'),
        },
        feature_description=cmumosei_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=use_multiple_images,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=output_features,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='imagechat', output_dir='./imagechat'),
      ],
    )
    TaskRegistry.add(
      f'{name}:metriconly',
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          'test': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                               'cmumosei-test.tfrecord*'),
        },
        feature_description=cmumosei_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=use_multiple_images,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=output_features,
      metric_fns=[
        mosei_exact_match,
      ],
    )
  else:
    raise NotImplementedError


def add_cmu_mosei_label(name, preprocess_fn):
  if "cmumosei" in name:
    _, mosei_key, version, label = name.split(':')
    if version == '1.0.0':
      use_multiple_images = True
      output_features = FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES
    elif version == '1.0.1':
      use_multiple_images = False
      output_features = FINETUNE_OUTPUT_FEATURES
    else:
      raise NotImplementedError

    TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
        split_to_filepattern={
          'train': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                                f'cmumosei-train-{mosei_key}-{label}.tfrecord*'),
          'validation': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                                     f'cmumosei-val-{mosei_key}-{label}.tfrecord*'),
          'test': os.path.join(TFDS_DATA_DIR, 'cmumosei', '1.0.0',
                               f'cmumosei-test-{mosei_key}-{label}.tfrecord*'),
        },
        feature_description=cmumosei_label_keys_to_features,
      ),
      preprocessors=[
        preprocess_fn,
        tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(video_to_text_preprocessor,
                          decode_jpeg=True,
                          use_multiple_images=use_multiple_images,
                          pass_through=('text_targets_pretokenized',
                                        'text_inputs_pretokenized',
                                        )),
      ],
      output_features=output_features,
      metric_fns=[
        perplexity,
        functools.partial(save_predictions, task='imagechat', output_dir='./imagechat'),
      ],
    )
  else:
    raise NotImplementedError


CMU_MOSEI_KEYS = ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']


def _unsparsify(x):
  if isinstance(x, tf.SparseTensor):
    x = x.values
  if x.dtype == tf.int64:
    x = tf.cast(x, dtype=tf.int32)
  return x


def get_response(x, mosei_key, word):
  response = tf.where(tf.strings.regex_full_match(x[mosei_key], '.*No.*'),
                      tf.constant(f'No, the person is not {word}.'),
                      tf.constant(f'Yes, the person is {word}.'))
  return response


def get_context_and_response(x, mosei_key):
  if mosei_key == 'sentiment':
    context = tf.strings.join(['context: ', x['transcript'], '. question: Is the person positive?'])
    response = x['sentiment']
  elif mosei_key in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
    if mosei_key == 'anger':
      context = tf.strings.join(['context: ', x['transcript'], f'. question: Is the person angry?'])
      response = get_response(x, mosei_key, 'angry')
    elif mosei_key == 'surprise':
      context = tf.strings.join(['context: ', x['transcript'], f'. question: Is the person surprised?'])
      response = get_response(x, mosei_key, 'surprised')
    elif mosei_key == 'disgust':
      context = tf.strings.join(['context: ', x['transcript'], f'. question: Is the person disgusted?'])
      response = get_response(x, mosei_key, 'disgusted')
    elif mosei_key == 'fear':
      context = tf.strings.join(['context: ', x['transcript'], f'. question: Is the person in fear?'])
      response = get_response(x, mosei_key, 'in fear')
    else:
      context = tf.strings.join(['context: ', x['transcript'], f'. question: Is the person {mosei_key}?'])
      response = x[mosei_key]
  else:
    raise NotImplementedError
  return tf.strings.lower(context), tf.strings.lower(response)


@seqio.map_over_dataset
def _preprocess_cmumosei_multi_images(x, mosei_key):
  images = tf.stack([x[f'image/{i}'] for i in range(3)])
  context, response = get_context_and_response(x, mosei_key)

  return {
    'image': images,
    'context': context,
    'response': response,
  }


@seqio.map_over_dataset
def _preprocess_cmumosei_multi_images_label(x, mosei_key):
  images = tf.stack([x[f'image/{i}'] for i in range(3)])
  x[mosei_key] = x['response']
  context, response = get_context_and_response(x, mosei_key)

  return {
    'image': images,
    'context': context,
    'response': response,
  }


@seqio.map_over_dataset
def _preprocess_cmumosei_single_image(x, mosei_key):
  images = tf.stack([x[f'image/{i}'] for i in range(3)])
  context, response = get_context_and_response(x, mosei_key)

  return {
    'image': images[0],
    'context': context,
    'response': response,
  }


@seqio.map_over_dataset
def _preprocess_cmumosei_single_image_label(x, mosei_key):
  images = tf.stack([x[f'image/{i}'] for i in range(3)])
  x[mosei_key] = x['response']
  context, response = get_context_and_response(x, mosei_key)

  return {
    'image': images[0],
    'context': context,
    'response': response,
  }


# mosei ==> 16,327 train samples

for mosei_key in CMU_MOSEI_KEYS:
  add_cmu_mosei(f'cmumosei:{mosei_key}:1.0.0',  # multi image
                functools.partial(_preprocess_cmumosei_multi_images, mosei_key=mosei_key))
  add_cmu_mosei(f'cmumosei:{mosei_key}:1.0.1',  # single image
                functools.partial(_preprocess_cmumosei_single_image, mosei_key=mosei_key))
