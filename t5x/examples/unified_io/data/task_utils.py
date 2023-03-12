import logging
import re
from collections import Counter
from typing import Sequence, Tuple, Union

import numpy as np
import seqio.preprocessors
import tensorflow as tf
import transformers

from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from t5x.examples.unified_io.evaluator import UnifiedIOOutput

TFDS_DATA_DIR = ''


# === Utility methods ===

def perplexity(targets: Sequence[str], scores: Sequence[int]):
  gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
  sentencepiece_tokenizer = get_default_vocabulary()
  gpt2_tokens = [gpt2_tokenizer(t)['input_ids'] for t in targets]
  sentencepiece_tokens = [sentencepiece_tokenizer.encode(t) for t in targets]
  normalized_scores = [s * len(t) / len(v) for s, t, v in zip(scores, sentencepiece_tokens, gpt2_tokens)]

  return {
    'perplexity': seqio.metrics.Scalar(np.exp(-1.0 * np.mean(scores))),
    'normalized_perplexity': seqio.metrics.Scalar(np.exp(-1.0 * np.mean(normalized_scores))),
  }


def dialogue_metrics(targets: Sequence[str], predictions: Sequence[Union[str, UnifiedIOOutput]]):
  """F1, BLEU, dist, and ROUGE metrics for dialogue."""
  re_art = re.compile(r'\b(a|an|the)\b')
  re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

  def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

  def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
      return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

  def _ngram(tokens, n: int):
    for i in range(len(tokens) - n + 1):
      yield tuple(tokens[i: i + n])

  f1_scores = []
  # bleu_scores = []
  # rouge_scores = []

  # ngram2_counter = Counter()
  ngram3_counter = Counter()
  # ngram4_counter = Counter()

  for target, prediction in zip(targets, predictions):
    if isinstance(predictions[0], UnifiedIOOutput):
      prediction_text = prediction.text
    else:
      prediction_text = prediction

    # f1 score
    target_unigram = normalize_answer(target).split(' ')
    prediction_unigram = normalize_answer(prediction_text).split(' ')
    p, r, f1 = _prec_recall_f1_score(prediction_unigram, target_unigram)
    f1_scores.append(f1)

    # bleu score
    k = 4  # bleu-4
    weights = [1 / k for _ in range(k)]
    # bleu_score = nltkbleu.sentence_bleu(
    #   [normalize_answer(a).split(" ") for a in [target]],
    #   normalize_answer(prediction_text).split(" "),
    #   smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    #   weights=weights,
    # )
    # bleu_scores.append(bleu_score)

    # rouge score
    # rouge_evaluator = rouge_scorer.RougeScorer(['rougeL'])
    # rouge_score = rouge_evaluator.score(normalize_answer(target), normalize_answer(prediction_text))['rougeL'].recall
    # rouge_scores.append(rouge_score)

    # dist score
    tokens = normalize_answer(prediction_text).split()
    # ngram2_counter.update(_ngram(tokens, 2))
    ngram3_counter.update(_ngram(tokens, 3))
    # ngram4_counter.update(_ngram(tokens, 4))

  # dist2_score = max(len(ngram2_counter), 1e-12) / max(sum(ngram2_counter.values()), 1e-5)
  dist3_score = max(len(ngram3_counter), 1e-12) / max(sum(ngram3_counter.values()), 1e-5)
  # dist4_score = max(len(ngram4_counter), 1e-12) / max(sum(ngram4_counter.values()), 1e-5)

  return {'f1': seqio.metrics.Scalar(np.mean(f1_scores)),
          # 'bleu4': seqio.metrics.Scalar(np.mean(bleu_scores)),
          # 'rouge': seqio.metrics.Scalar(np.mean(rouge_scores)),
          # 'dist2': seqio.metrics.Scalar(dist2_score),
          'dist3': seqio.metrics.Scalar(dist3_score)}
          # 'dist4': seqio.metrics.Scalar(dist4_score)}


def store_scores(targets: Sequence[Tuple[str, str]],
                 scores: Sequence[int]):
  output = []
  logging.info(f"Start saving scores.")
  for (image_id, round_id, input, target), score in zip(targets, scores):
    d = {'image_id': int(image_id),
         'round_id': int(round_id),
         'input': input,
         'target': target,
         'score': score}
    if len(output) < 2:
      print(d)
    output.append(d)

  return {'dummy': output}


def store_scores_vcr(targets: Sequence[Tuple[str, str]],
                     scores: Sequence[int]):
  output = []
  logging.info(f"Start saving scores.")
  for (annot_id, answer_label, rationale_label, text_target, text_input), score in zip(targets, scores):
    d = {"annot_id": annot_id,
         "answer_label": int(answer_label),
         "rationale_label": int(rationale_label),
         "input": text_input,
         "target": text_target,
         "score": score}
    if len(output) < 2:
      print(d)
    output.append(d)

  return {'dummy': output}


# === keys to features ===
ytdialogue_keys_to_features = {
  'youtube_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'start_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'end_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

ytdialogue_multiple_images_keys_to_features = {
  'youtube_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'start_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'end_sec': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'num_turns': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}
for i in range(16):
  ytdialogue_multiple_images_keys_to_features[f'image/{i}'] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)

imagechat_keys_to_features = {
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'caption': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}
visdial_keys_to_features = {
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'caption': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}
visdial_dense_keys_to_features = {
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'caption': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'score': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
}
visdial_dense_pair_keys_to_features = {
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'positive_response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'negative_response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'caption': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}
visdial_ndcg_keys_to_features = {
  'context': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'caption': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image_id': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
  'round_id': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
}
cmumosei_keys_to_features = {
  'sentiment': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'happy': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'sad': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'anger': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'surprise': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'disgust': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'fear': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/0': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/1': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/2': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

cmumosei_label_keys_to_features = {
  'response': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'transcript': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/0': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/1': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'image/2': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}
