import seqio
import tensorflow as tf

from t5x.examples.unified_io.data.data_utils import get_default_vocabulary


def process_strings(strings):
  first_char = tf.strings.substr(strings, 0, 1)
  uppercased_first_char = tf.strings.upper(first_char)
  rest_of_string = tf.strings.substr(strings, 1, -1)
  processed_strings = uppercased_first_char + rest_of_string

  # add the '?' character to the end of each string only for odd indices
  indices = tf.range(tf.shape(strings)[0])
  add_question_mark = tf.math.equal(tf.math.mod(indices, 2), 0)
  add_period = tf.math.equal(tf.math.mod(indices, 2), 1)
  processed_strings = tf.where(add_question_mark, processed_strings + "?", processed_strings)
  processed_strings = tf.where(add_period, processed_strings + ".", processed_strings)

  return processed_strings


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

  meta = tf.concat([voc.encode_tf('<extra_id_1>'), voc.encode_tf(x['meta'])], axis=0)

  context = tf.strings.split(x['context'], '\n')
  context = tf.strings.strip(context)
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
  out['text_inputs_pretokenized'] = tf.strings.join([x['meta'], x['context']], separator='\t')
  context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
  out['text_inputs'] = tf.concat([meta, context_tokenized.values], axis=0)
  out['image'] = x['image']
  return out


@seqio.map_over_dataset
def tokenize_visdial(x):
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
  context = process_strings(context)
  turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
  out['text_inputs_pretokenized'] = tf.strings.join([x['meta'], x['context']], separator='\t')
  context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
  out['text_inputs'] = tf.concat([meta, context_tokenized.values], axis=0)
  out['image'] = x['image']
  return out


@seqio.map_over_dataset
def tokenize_ndcg(x, no_context=False):
  """
  Let's assume that dialogue data has the context and the response.
  context should be the list of string, separated by turns.
  """
  voc = get_default_vocabulary()

  out = {
    'image_id': x['image_id'],
    'round_id': x['round_id'],
  }
  if x['response'].dtype == tf.string:
    out['text_targets_pretokenized'] = x['response']
    out['text_targets'] = voc.encode_tf(tf.strings.strip(x['response']))

  if not no_context:
    meta = tf.concat([voc.encode_tf('<extra_id_1>'), voc.encode_tf(x['meta'])], axis=0)

    context = tf.strings.split(x['context'], '\n')
    context = tf.strings.strip(context)
    context = process_strings(context)
    turn_tokens = tf.tile(tf.reshape(voc.encode_tf('<extra_id_0>'), [1, 1]), [len(context), 1])
    out['text_inputs_pretokenized'] = tf.strings.join([x['meta'], x['context']], separator='\t')
    context_tokenized = tf.concat([turn_tokens, voc.encode_tf(context)], axis=1)
    out['text_inputs'] = tf.concat([meta, context_tokenized.values], axis=0)
  else:
    out['text_inputs_pretokenized'] = tf.constant('?', dtype=tf.string)
    out['text_inputs'] = voc.encode_tf(out['text_inputs_pretokenized'])

  out['image'] = x['image']
  return out
