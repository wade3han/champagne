import functools

import seqio.preprocessors
import tensorflow as tf
from seqio import TaskRegistry, MixtureRegistry

from t5x.examples.unified_io.data import multi_images_visual_dialogue_tasks  # noqa
from t5x.examples.unified_io.data import visual_dialogue_meta_tasks  # noqa
from .data_utils import get_default_vocabulary
from .metrics import save_predictions, save_predictions_dialogue
from .postprocessors import dialogue_postprocessor, dialogue_postprocessor_infer
from .preprocessors import text_to_text_preprocessor
from .task_constants import FINETUNE_OUTPUT_FEATURES, FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES
from .task_utils import TFDS_DATA_DIR, perplexity, dialogue_metrics

"""
This file registers datasets related to dialogue. 
"""


# === Utility methods ===

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
    out['text_targets'] = voc.encode_tf(x['response'])

  context = tf.strings.split(x['context'], '\n')
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
  out['text_inputs_pretokenized'] = x['context']
  context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
  out['text_inputs'] = context_tokenized.values
  return out


# ==== Text-only dialogue ====
TEXT_ONLY_DIALOGUE_TASKS = []
TEXT_ONLY_MI_DIALOGUE_TASKS = []
TEXT_ONLY_DIALOGUE_TASKS_METRIC_ONLY = []
TEXT_ONLY_MI_DIALOGUE_TASKS_METRIC_ONLY = []


def add_text_only_dialogue(name, preprocess_fn):
  TEXT_ONLY_DIALOGUE_TASKS.append(f't2t_dialogue___{name}')
  TEXT_ONLY_DIALOGUE_TASKS_METRIC_ONLY.append(f't2t_dialogue___{name}:metriconly')

  TaskRegistry.add(
    f't2t_dialogue___{name}',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized')),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[
      perplexity,
      functools.partial(save_predictions, task='name', output_dir='./t2t_dialogue'),
    ],
  )

  TaskRegistry.add(
    f't2t_dialogue___{name}:metriconly',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized')),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=dialogue_postprocessor,
    metric_fns=[
      perplexity,
      dialogue_metrics,
    ],
  )

  TaskRegistry.add(
    f't2t_dialogue___{name}:test',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized')),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=dialogue_postprocessor_infer,
    metric_fns=[
      functools.partial(save_predictions_dialogue, output_dir=name),
    ],
  )

  TEXT_ONLY_MI_DIALOGUE_TASKS.append(f't2t_dialogue_mi___{name}')
  TEXT_ONLY_MI_DIALOGUE_TASKS_METRIC_ONLY.append(f't2t_dialogue_mi___{name}:metriconly')

  TaskRegistry.add(
    f't2t_dialogue_mi___{name}',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized'),
                        use_multiple_images=True),
    ],
    output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
    metric_fns=[
      perplexity,
      functools.partial(save_predictions, task='t2t_dialogue', output_dir='./t2t_dialogue'),
    ],
  )

  TaskRegistry.add(
    f't2t_dialogue_mi___{name}:metriconly',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized'),
                        use_multiple_images=True),
    ],
    output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
    postprocess_fn=dialogue_postprocessor,
    metric_fns=[
      perplexity,
      dialogue_metrics,
    ],
  )
  TaskRegistry.add(
    f't2t_dialogue_mi___{name}:test',
    source=seqio.TfdsDataSource(
      tfds_name=f'{name}:1.0.0',
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(text_to_text_preprocessor,
                        pass_through=('text_targets_pretokenized',
                                      'text_inputs_pretokenized'),
                        use_multiple_images=True),
    ],
    output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
    postprocess_fn=dialogue_postprocessor_infer,
    metric_fns=[
      functools.partial(save_predictions_dialogue, output_dir=name),
    ],
  )


@seqio.map_over_dataset
def _preprocess_bst(x):
  return {
    'context': x['meta'] + '\n' + x['context'],
    # 'meta': x['meta'],
    # 'context': x['context'],
    'response': x['response'],
  }


@seqio.map_over_dataset
def _preprocess_convai(x):
  return {
    'context': x['meta'] + '\n' + x['context'],
    # 'meta': x['meta'],
    # 'context': x['context'],
    'response': x['response'],
  }


@seqio.map_over_dataset
def _preprocess_wow(x):
  return {
    'context': x['meta'] + '\n' + x['context'],
    # 'meta': x['meta'],
    # 'context': x['context'],
    'response': x['response'],
  }


@seqio.map_over_dataset
def _preprocess_ed(x):
  return {
    'context': x['meta'] + '\n' + x['context'],
    # 'meta': x['meta'],
    # 'context': x['context'],
    'response': x['response'],
  }


@seqio.map_over_dataset
def _preprocess_woi(x):
  return {
    'context': x['meta'] + '\n' + x['context'],
    # 'meta': x['meta'],
    # 'context': x['context'],
    'response': x['response'],
  }


### DATA SAMPLE NUMBER for TRAIN SET
# bst ==> 26,947
# convai ==> 104,771
# wow ==> 46,613
# ed ==> 25,678
# woi ==> 37,508
# ic ==> 169,300, 661 steps for 1 epoch if bs = 256
# mix total == 410,817 samples, 1604 steps for 1 epoch if bs = 256


add_text_only_dialogue('bst', _preprocess_bst)  # 7k conversations; 4,819 train / 1,009 valid / 980 test
add_text_only_dialogue('convai', _preprocess_convai)  # 17,878 train / 1,000 valid / 1,000 test
add_text_only_dialogue('wow', _preprocess_wow)  # 18,430 train / 981 valid / 965 test
add_text_only_dialogue('ed', _preprocess_ed)  # 39,057 train / 2,769 valid / 2,547 test
add_text_only_dialogue('woi', _preprocess_woi)  # 8,614 train / 516 valid / 503 test

# visdial has 123,287 images x 10 rounds, so the weight should be lower to balance the training.
TEXT_ONLY_DIALOGUE_TASKS_BALANCED = [
  (name, 1.0) if name == 't2t_dialogue___bst' else (name, 3.0) for name in TEXT_ONLY_DIALOGUE_TASKS
]
MixtureRegistry.add('dialogue_mix:balanced:meta_firstturn',
                    TEXT_ONLY_DIALOGUE_TASKS_BALANCED + [('imagechat:1.1.0:include_firstturn', 3.0)],
                    default_rate=3.0)

TEXT_ONLY_MI_DIALOGUE_TASKS_BALANCED = [
  (name, 1.0) if name == 't2t_dialogue_mi___bst' else (name, 3.0) for name in TEXT_ONLY_MI_DIALOGUE_TASKS
]
MixtureRegistry.add('dialogue_mi_mix:balanced:meta_firstturn',
                    TEXT_ONLY_MI_DIALOGUE_TASKS_BALANCED + [('imagechat:1.2.0:include_firstturn', 3.0)],
                    default_rate=3.0)
