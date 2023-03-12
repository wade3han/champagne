import functools
import json
import logging
import re
from collections import Counter, defaultdict
from os.path import join

import gin
import seqio
import seqio.preprocessors
from seqio.loggers import Logger

from scipy.stats import spearmanr
from seqio import FileDataSource, TaskRegistry, MixtureRegistry


from seqio.metrics import Scalar, Text
from .metrics import exact_match
from .preprocessors import text_to_text_preprocessor, read_input_target_tsv, rekey
from .task_constants import *

from typing import Sequence, Mapping, Optional, Any
import numpy as np

from t5x.examples.unified_io.metrics.squad_official_evaluation import compute_exact as squad_exact_match
from t5x.examples.unified_io.metrics.squad_official_evaluation import compute_f1 as squad_f1

"""
This file registers NLP (text-to-text) tasks. It has 5 Mixtures:

- glue (Tasks in GLUE)
- super_glue  (Tasks in SuperGLUE)
- sentence_classification  (Other sentence classification tasks)
- extractive_qa (extractive QA tasks, we use SQuAD 2.0 and other tasks from MRQ, 
we use a sliding-window approach to handle long documents)
- mc_qa (multiple choice QA)

ad well the gigaword summerization dataset.

Metrics are provided for all tasks (TODO besides summarization), 
as well a Loggers that can generate official submissions to GLUE or SuerGLUE.

All tasks source from the public tensorflow dataset catalog  
"""



USE_V1_PROMPTS = False
"""Use prompts from UNICORN, Unified-IO v1 (for the first arXiv paper) was trained on these"""

SEPERATE_SENTENCES = False if USE_V1_PROMPTS else True
"""For multi-part input, should we have a header for each part or not"""


# === General pre-processing methods, we try to use this whenever applicable for consistency ===

def build_multiple_choice_question(question, choices):
  parts = ["question:", question]
  if isinstance(choices, list):
    for i, c in enumerate(choices):
      parts += [f"choice{i+1}:", c]
  else:
    # Handle variable length tensors
    n = tf.shape(choices)[0]
    prefix1 = tf.fill((n, ), "choice")
    prefix2 = tf.as_string(tf.range(1, n+1))
    prefix3 = tf.fill((n, ), ": ")
    choices = tf.stack([prefix1, prefix2, prefix3, choices], 1)
    choices = tf.strings.reduce_join(choices, 1)
    parts += [tf.strings.reduce_join(choices, 0, separator=" ")]

  return tf.strings.join(parts, " ")


def build_context_from_sentences(sent1, sent2=None):
  context = [
    "sentence1: " if SEPERATE_SENTENCES else "context: ",
    _fix_whitespace(sent1)
  ]
  if sent2 is not None:
    context += [
      " sentence2: " if SEPERATE_SENTENCES else " ",
      _fix_whitespace(sent2),
    ]
  return tf.strings.join(context)


# === Utility methods ===


def nlp_post_processor(output_or_target, example=None, is_target=False):
  if is_target:
    out = dict(text_target=output_or_target)
    # Meta-data we sometimes need for evaluation
    for k in ["label", "example_id", "answers", "choices"]:
      if k in example:
        out[k] = example[k]
    return out
  else:
    return output_or_target


def _fix_whitespace(x):
  # Remove double/extra white space in `x`
  # \u00A0 is no-break space we replace with regular spaces
  return tf.strings.strip(tf.strings.regex_replace(x, "[ \u00A0]+", " "))


@seqio.map_over_dataset
def tokenize(x):
  voc = get_default_vocabulary()

  out = dict(x)
  if out["text_targets"].dtype == tf.string:
    out["text_targets_pretokenized"] = x["text_targets"]
    out["text_targets"] = voc.encode_tf(x["text_targets"])

  input_keys = ["question", "context"]
  if all(out[k].dtype == tf.string for k in input_keys):
    parts = [x["context"], x["question"]]
    # The tokenizer automatically adds a leading space. so we need to join with a
    # space to be equal to concatenating the tokenized inputs
    out["text_inputs_pretokenized"] = tf.strings.join(parts, " ")
    for key in ["question", "context"]:
        out[key] = voc.encode_tf(x[key])
  return out


@seqio.map_over_dataset
def build_text_inputs(x, sequence_length):
  """Build the input from the question and context dataset fields

  Ensures only the context will get truncated if we have the max input length
  """
  max_input_len = sequence_length["text_inputs"]

  context = x["context"]
  question = x["question"]
  n = max_input_len - tf.shape(question)[0]
  text_inputs = tf.concat([context[:n], question], 0)

  out = {k: v for k, v in x.items() if k not in ["question", "context"]}
  out["text_inputs"] = text_inputs
  return out


@seqio.map_over_dataset
def build_sentence_classification(
    example, output_features, sent1_key, sent2_key, prompt, label_mapping, label_buckets=None):
  label_mapping = tf.constant(label_mapping)
  if label_buckets is not None:
    label_buckets = tf.constant(label_buckets, dtype=tf.float32)
  vocab = output_features["text_inputs"].vocabulary

  # We build the context and question # separately so if the input is longer then
  # our max sequence length, we can truncate the context without cutting off the question.
  context = build_context_from_sentences(
    example[sent1_key], None if sent2_key is None else example[sent2_key])
  question = tf.strings.join([" question: ", prompt])

  label = example["label"]
  if label_buckets is not None:
    label_ix = tf.reduce_sum(tf.cast(label >= label_buckets, tf.int32))
  else:
    label_ix = label

  out = dict(
    question=vocab.encode_tf(question),
    context=vocab.encode_tf(context),
    text_targets=vocab.encode_tf(label_mapping[label_ix]),
    pretokenized_prompt=tf.strings.join([context, question]),
    label=example["label"]
  )
  if "idx" in example:  # Used in SuperGLUE/GLUE
    out["idx"] = example["idx"]
  return out


def add_sentence_classification(dataset_name, tfds_name, metrics, prompt, sent1_key, sent2_key, splits=None):
  if len(prompt) == 3:
    prompt, label_mapping, buckets = prompt
  else:
    buckets = None
    prompt, label_mapping = prompt

  TaskRegistry.add(
    dataset_name,
    source=seqio.TfdsDataSource(
      tfds_name=tfds_name,
      tfds_data_dir=TFDS_DATA_DIR,
      splits=splits
    ),
    preprocessors=[
      functools.partial(build_sentence_classification, sent1_key=sent1_key, sent2_key=sent2_key,
                        prompt=prompt, label_mapping=label_mapping, label_buckets=buckets),
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      build_text_inputs,
      functools.partial(text_to_text_preprocessor,
                        pass_through=("example_id", "label", "pretokenized_text_inputs")),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=nlp_post_processor,
    metric_fns=metrics
  )


# === GLUE ===

ENTAILMENT_LABELS = ['entailment', 'neutral', 'contradiction']

# Prompts from Unicorn we used in v1.0 for reference
UNICORN_GLUE_PROMPTS = {
  "cola": ("Is the sentence acceptable or unacceptable?", ["unacceptable", "acceptable"]),
  "sst2": ("Is this negative or positive?", ["negative", "positive"]),
  "mrpc": ("Are these sentences equivalent or not equivalent?",
           ["No, they are not equivalent.", "Yes, they are equivalent."]),
  "qqp": ("Are these sentences duplicate or not?",
          ["No, they are not duplicate.", "Yes, they are duplicate."]),
  "stsb": ("How similar or different are the meaning of these sentences?",
           ["very different", "somewhat different", "neither",
            "somewhat similar", "very similar"],
           [0.75, 2.55, 3.4, 4.15, 5.01]
           ),
  "mnli": ("Is the relation of the sentences entailment, neutral or contradiction?",
           ENTAILMENT_LABELS),
  "qnli": ("Does the first sentence entail the second sentence?",
           ["Yes, it entails.", "No, it does not entail."]),
  "rte": ("Does this sentence entail the following sentence?",
          ["Yes, it entails.", "No, it does not entail."]),
  "wnli": ("Does the first sentence entail the second sentence?",
           ["No, it does not entail.", "Yes, it entails."]),
}

# Revised prompts we use instead
OUR_GLUE_PROMPTS = {
  "cola": ("Is the sentence grammatically acceptable?", ["no", "yes"]),
  "sst2": ("Does this sentence have a negative or positive sentiment?", ["negative", "positive"]),
  "mrpc": ("Are these sentences equivalent?", ["no", "yes"]),
  "qqp": ("Are these questions equivalent?", ["no", "yes"]),
  "stsb": ("How similar or different are the meaning of these sentences?",
           ["very different", "somewhat different", "neither",
            "somewhat similar", "very similar"],
           [0.75, 2.55, 3.4, 4.15, 5.01]
           ),
  "mnli": ("Is the the second sentence an entailment, neutral or contradiction or the first sentence?",
           ENTAILMENT_LABELS),
  "qnli": ("Does the second sentence contain the answer to the question?",
           ["yes", "no"]),
  "rte": ("Does the first sentence entail the second sentence?",
          ["yes", "no"]),
  "wnli": ("Does the first sentence entail the second sentence?",
           ["no", "yes"]),
  "cb": ("Does the first sentence entail the second sentence?",
           ["no", "yes"]),
}

GLUE_PROMPTS = UNICORN_GLUE_PROMPTS if USE_V1_PROMPTS else OUR_GLUE_PROMPTS


COLA_LABELS = {
  "acceptable": 1,
  "unacceptable": -1
}


def cola_matthews_cor(targets: Sequence, predictions: Sequence[str], aux_values):
  from sklearn.metrics import matthews_corrcoef
  if all(t["text_target"] == "none" for t in targets):
    # Test set has `none` labels
    return {}
  target_scores = np.array([COLA_LABELS[t["text_target"]] for t in targets])
  pred_scores = np.array([COLA_LABELS.get(p, 1) for p in predictions])
  corr = matthews_corrcoef(target_scores, pred_scores)
  return {
    "matthews": Scalar(corr),
  }


STSB_LABELS = {k: i + 1 for i, k in enumerate(GLUE_PROMPTS["stsb"][1])}


def stsb_cor(targets: Sequence, predictions: Sequence[str], aux_values):
  mapping = {name: val for name, val in zip(*GLUE_PROMPTS["stsb"][1:])}
  if all(t["text_target"] == "none" for t in targets):
    # Test set has `none` labels
    return {}
  target_scores = np.array([t["label"] for t in targets])
  pred_scores = np.array([mapping.get(p.lower(), 3) for p in predictions])
  corr, _ = spearmanr(target_scores, pred_scores)
  return {
    "spearmanr": Scalar(corr),
  }


def f1_match(targets: Sequence, predictions: Sequence[str], aux_values, positive_label):
  label_positive = np.array([positive_label == target["text_target"] for target in targets])
  pred_positive = np.array([pred.lower().strip() == positive_label.lower() for pred in predictions])
  true_positive = np.logical_and(pred_positive, label_positive).sum()
  false_positive = np.logical_and(pred_positive, np.logical_not(label_positive)).sum()
  false_negative = np.logical_and(np.logical_not(pred_positive), label_positive).sum()
  f1 = true_positive / (true_positive + 0.5 * (false_positive + false_negative))
  return {
    "f1": Scalar(f1)
  }


# Metadata we need to build official GLUE TSVs,
# Includes class names, default output class, and file name
GLUE_TSV_LABELS = {
  "cola": ([0, 1], 1, "CoLA"),
  "sst2": ([0, 1], 1, "SST-2"),
  "mrpc": ([0, 1], 1, "MRPC"),
  "qqp": ([0, 1], 0, "QQP"),
  "stsb": ([0.5, 2, 3, 4, 5], 2.7, "STS-B"),
  "mnli": (ENTAILMENT_LABELS, 'neutral', "MNLI-m"),
  "mnli_mismatched": (ENTAILMENT_LABELS, 'neutral', "MNLI-mm"),
  "ax": (ENTAILMENT_LABELS, 'neutral', "AX"),
  "qnli": (["entailment", "not_entailment"], "entailment", "QNLI"),
  "rte": (["entailment", "not_entailment"], "entailment", "RTE"),
  "wnli": ([0, 1], 0, "WNLI"),
}


@gin.register(denylist=["output_dir"])
class SaveGlue(Logger):
  """Saves GLUE TSV files that can be submitted to the leaderboard"""

  def __call__(self, task_name: str, step: Optional[int],
               metrics: Mapping[str, Any],
               dataset: Optional[tf.data.Dataset],
               inferences: Optional[Mapping[str, Sequence[Any]]],
               targets: Optional[Sequence[Any]]) -> None:
    if not task_name.startswith("glue_"):
      return
    glue_name = re.split("_+", task_name, maxsplit=1)[-1]
    if glue_name.startswith("qqp_"):
      glue_name = "qqp"
    invalid_outputs = []
    out = {}
    tsv_classes, tsv_default, tsv_name = GLUE_TSV_LABELS[glue_name]

    if glue_name == "mnli_mismatched" or glue_name == "ax":
      prompt_classes = GLUE_PROMPTS["mnli"]
    else:
      prompt_classes = GLUE_PROMPTS[glue_name]
    if len(prompt_classes) == 3:
      prompt_labels, prompt_buckets = prompt_classes[1:]
    else:
      prompt_labels, prompt_buckets = prompt_classes[1], None
    prompt_labels = {k: i for i, k in enumerate(prompt_labels)}

    for output, target in zip(inferences["prediction"], targets):
      example_id = target["example_id"]
      if example_id is None:
        raise ValueError()
      example_id = str(example_id)
      if example_id in out:
        raise ValueError()

      # Map output to the TSV class name we need to submit with
      if output not in prompt_labels:
        tsv_pred = tsv_default
        invalid_outputs.append(output)
      else:
        ix = prompt_labels[output]
        tsv_pred = tsv_classes[ix]
      out[example_id] = str(tsv_pred)

    output_filename = join(self.output_dir, f"glue/{tsv_name}.tsv")
    if invalid_outputs:
      logging.info(f"{len(invalid_outputs)} ({len(invalid_outputs)/len(targets):0.2f}) invalid predictions")
      for w, c in Counter(invalid_outputs).most_common(10):
        logging.info(f"Invalid output: {w} n_occurances={c}")
    logging.info(f"Saving {glue_name} GLUE predictions to {output_filename}")
    with tf.io.gfile.GFile(output_filename, "w") as f:
      f.write(f"ID\tClass\n")  # GlUE submission must have a header
      for k, v in out.items():
        f.write(f"{k}\t{v}\n")


SUPER_GLUE_JSON_LABELS = {
  # Tasks with binary labels
  "multirc": ([0, 1], 0, "MultiRC"),
  "boolq": (["false", "true"], "true", "BoolQ"),
  "wic": (["false", "true"], "false", "WiC"),
  "wsc.fixed": (["False", "True"], "False", "WSC"),
  "rte": (["not_entailment", "entailment"], "not_entailment", "RTE"),
  "axb": (["not_entailment", "entailment"], "not_entailment", "AX-b"),
  "axg": (["not_entailment", "entailment"], "not_entailment", "AX-g"),

  # Special cases
  "record": (None, None, "ReCoRD"),
  "cb": (None, None, "CB"),
  "copa": (None, None, "COPA"),
}


@gin.register(denylist=["output_dir"])
class SaveSuperGlue(Logger):
  """Saves SuperGLUE JSONL files that can be submitted to the leaderboard"""

  def __call__(self, task_name: str, step: Optional[int],
               metrics: Mapping[str, Any],
               dataset: Optional[tf.data.Dataset],
               inferences: Optional[Mapping[str, Sequence[Any]]],
               targets: Optional[Sequence[Any]]) -> None:
    if not task_name.startswith("super_glue"):
      return
    glue_name = re.split("_+", task_name, maxsplit=2)[-1]
    binary_labels, default_value, file_name = SUPER_GLUE_JSON_LABELS[glue_name]
    output_filename = join(self.output_dir, f"super_glue/{file_name}.jsonl")

    out = []
    invalid_outputs = []
    for output, target in zip(inferences["prediction"], targets):

      if glue_name == "record":
        # record is multiple choice, but output is the text of the choice
        choices = [x.decode("utf-8") for x in target["choices"]]
        if output not in choices:
          invalid_outputs.append(output)
          output = choices[0]

      elif glue_name == "copa":
        # COPA is multiple choice where we must return the choice index
        choices = [x.decode("utf-8") for x in target["choices"]]
        try:
          ix = choices.index(output)
          output = ix
        except ValueError:
          # TODO could try to find the closet choice
          invalid_outputs.append(output)
          output = 0

      elif glue_name == "cb":
        # CB requires MNLI entailment labels
        if output not in ENTAILMENT_LABELS:
          invalid_outputs.append(output)
          output = "contradiction"

      else:
        output = output.lower()
        # This is a bit hacky, but we detect the binary class based on some standard
        # ways of phrasing a positive/negative answer
        if output.startswith("yes") or output.startswith("true") or output == "entailment":
          output = binary_labels[1]
        elif output.startswith("no") or output.startswith("false"):
          output = binary_labels[0]
        else:
          invalid_outputs.append(output)
          output = default_value

      example_id = target["example_id"]
      if glue_name != "multirc":
        if isinstance(example_id, bytes):
          example_id = example_id.decode("utf-8")
        example_id = int(example_id)  # Make sure this is a python int
        out.append(dict(idx=example_id, label=output))
      else:
        example_id = target["example_id"].decode("utf-8")
        parts = example_id.split("-")
        paragraph_ix = int(parts[1])
        question_ix = int(parts[3])
        answer_ix = int(parts[5])
        out.append((paragraph_ix, question_ix, answer_ix, output))

    if glue_name == "multirc":
      grouped = defaultdict(lambda: defaultdict(list))
      for (p_ix, q_ix, a_ix, label) in out:
        grouped[p_ix][q_ix].append(dict(idx=a_ix, label=label))
      out = []
      for p_ix, qs in grouped.items():
        q_list = []
        for q_ix, ans in qs.items():
          q_list.append(dict(idx=q_ix, answers=ans))
        out.append(dict(
          idx=p_ix,
          passage=dict(questions=q_list)
        ))

    if invalid_outputs:
      logging.info(f"{len(invalid_outputs)} ({len(invalid_outputs)/len(targets):0.2f}) invalid predictions")
      for w, c in Counter(invalid_outputs).most_common(10):
        logging.info(f"Invalid output: {w} n_occurances={c}")
    logging.info(f"Saving {glue_name} SuperGLUE predictions to {output_filename}")
    with tf.io.gfile.GFile(output_filename, "w") as f:
      for val in out:
        f.write(json.dumps(val))
        f.write("\n")


GLUE_TASKS = []


def add_glue(name, sent1_key, sent2_key, splits=None):
  GLUE_TASKS.append(f"glue___{name}")

  if name == "ax" or "mnli" in name:
    prompt_name = "mnli"
  elif name.startswith("qqp_"):
    prompt_name = "qqp"
  else:
    prompt_name = name
  prompt = GLUE_PROMPTS[prompt_name]
  tfds_name = f"glue/{name}:2.0.0"

  metric_fns = [exact_match]
  if prompt_name == "qqp":
    metric_fns.append(functools.partial(f1_match, positive_label=prompt[1][-1]))
  elif prompt_name == "mrpc":
    metric_fns.append(functools.partial(f1_match, positive_label=prompt[1][-1]))
  elif prompt_name == "cola":
    metric_fns.append(cola_matthews_cor)
  elif prompt_name == "stsb":
    metric_fns.append(stsb_cor)
  add_sentence_classification(f"glue___{name}", tfds_name, metric_fns, prompt, sent1_key, sent2_key, splits)


add_glue("cola", "sentence", None)
add_glue("sst2", "sentence", None)
add_glue("mrpc", "sentence1", "sentence2")
add_glue("qqp", "question1", "question2")
add_glue("stsb", "sentence1", "sentence2")
add_glue("mnli", "premise", "hypothesis",
         {"train": "train", "validation": "validation_matched", "test": "test_matched"})
add_glue("mnli_mismatched", "premise", "hypothesis")
add_glue("qnli", "question", "sentence")
add_glue("rte", "sentence1", "sentence2")
add_glue("wnli", "sentence1", "sentence2")
add_glue("ax", "premise", "hypothesis")

MixtureRegistry.add("glue", GLUE_TASKS, default_rate=1.0)


# ==== SuperGlUE ====

SUPER_GLUE_TASKS = []


def add_superglue(name, preprocess_fn):
  SUPER_GLUE_TASKS.append(f"super_glue___{name}")

  TaskRegistry.add(
    f"super_glue___{name}",
    source=seqio.TfdsDataSource(
      tfds_name=f"super_glue/{name}:1.0.2",
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      build_text_inputs,
      functools.partial(text_to_text_preprocessor,
                        pass_through=("example_id", "label", "text_targets_pretokenized",
                                      "text_inputs_pretokenized", "choices")),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=nlp_post_processor,
    metric_fns=[exact_match]
  )


@seqio.map_over_dataset
def _preprocess_copa(x):
  if USE_V1_PROMPTS:
    context = tf.strings.join([
      "question: ", x["question"],
      " context: ", x["premise"]
    ])
    question = tf.strings.join([
      " question: which choice does the context relate to?",
      " context1: : ", x["choice1"],
      " choice2: ", x["choice2"],
    ])
  else:
    context = tf.strings.join(["context: ", x["premise"]])
    q = "What happened as a result?" if x["question"] == "effect" else "What was the cause of this?"
    question = build_multiple_choice_question(q, [x["choice1"], x["choice2"]])
  return dict(
    question=question,
    context=context,
    example_id=x["idx"],
    text_targets=x["choice1"] if x["label"] == 0 else x["choice2"],
    choices=[x["choice1"], x["choice2"]]
  )


@seqio.map_over_dataset
def _preprocess_multirc(x):
  voc = get_default_vocabulary()
  if USE_V1_PROMPTS:
    answers = ["False", "True"]
  else:
    answers = ["no", "yes"]
  answers = voc.encode_tf(tf.constant(answers))

  context = tf.strings.join(["context: ", x["paragraph"]])
  if USE_V1_PROMPTS:
    question = tf.strings.join([
      x["question"], x["answer"],
      "question: Is this answer true or false for the question?"
    ], " ")
  else:
    question = tf.strings.join([
      "question:", x["question"],
      " is ", x["answer"], " a correct answer?"
      # "answer:", x["answer"],
      # "question: Is this answer correct?"
    ], " ")
  idx = x["idx"]
  example_id = tf.strings.join([
    "paragraph-", tf.strings.as_string(idx["paragraph"]),
    "-question-", tf.strings.as_string(idx["question"]),
    "-answer-", tf.strings.as_string(idx["answer"])
  ])
  return dict(
    question=voc.encode_tf(question),
    context=voc.encode_tf(context),
    example_id=example_id,
    text_targets=answers[x["label"]],
  )


@seqio.map_over_dataset
def _preprocess_boolq(x):
  voc = get_default_vocabulary()
  if USE_V1_PROMPTS:
    answers = ["False", "True"]
  else:
    answers = ["no", "yes"]
  answers = voc.encode_tf(tf.constant(answers))

  return dict(
    question=tf.strings.join(["question: ", x["question"], "?"]),
    context=tf.strings.join(["context: ", x["passage"]]),
    example_id=x["idx"],
    text_targets=answers[x["label"]]
  )


@seqio.map_over_dataset
def _preprocess_wsc(x):
  voc = get_default_vocabulary()
  if USE_V1_PROMPTS:
    answers = ["No, they're different people.", "Yes, they are the same person."]
  else:
    answers = ["no", "yes"]
  answers = voc.encode_tf(tf.constant(answers))

  context = tf.strings.join(["context: ", x["text"]])
  if USE_V1_PROMPTS:
    question = tf.strings.join([
      "question: Are ",
      x["span1_text"],
      " and ",
      x["span2_text"],
      " the same person?"
    ])
  else:
    question = tf.strings.join([
      "question: Does \"",
      x["span1_text"],
      "\" and \"",
      x["span2_text"],
      "\" refer to the same person?"
    ])
  return dict(
    question=voc.encode_tf(question),
    context=voc.encode_tf(context),
    example_id=x["idx"],
    text_targets=answers[x["label"]]
  )


@seqio.map_over_dataset
def _preprocess_wic(x):
  voc = get_default_vocabulary()
  if USE_V1_PROMPTS:
    answers = ["False", "True"]
  else:
    answers = ["no", "yes"]
  answers = voc.encode_tf(tf.constant(answers))

  context = build_context_from_sentences(x["sentence1"], x["sentence2"])
  if USE_V1_PROMPTS:
    question = tf.strings.join([
      "question: Is the word ",
      x["word"],
      " used in the same sense in sentence1 and sentence2?"
    ])
  else:
    question = tf.strings.join([
      "question: Is the word \"",
      x["word"],
      "\" used in the same sense in both sentences?"
    ])
  return dict(
    question=voc.encode_tf(question),
    context=voc.encode_tf(context),
    example_id=x["idx"],
    text_targets=answers[x["label"]]
  )


@seqio.map_over_dataset
def _preprocess_record(x):
  context = tf.strings.join(["context: ", x["passage"]])
  if USE_V1_PROMPTS:
    question = tf.strings.join([
      "question:", x["query"],
      "entities:", tf.strings.reduce_join(x["entities"], separator=","),
      "question: Which named entity would fit in @placeholder?",
    ], " ")
  else:
    question = tf.strings.join([
      "sentence:", x["query"],
      "entities:", tf.strings.reduce_join(x["entities"], separator=", "),
      "question: Which named entity would fit in @placeholder?",
    ], " ")

  return dict(
    question=question,
    context=context,
    example_id=x["idx"]["query"],
    text_targets=x["answers"],
    choices=x["entities"]
  )


if USE_V1_PROMPTS:

  @seqio.map_over_dataset
  def _preprocess_rte(x):
    _labels = tf.constant(["entailment", "not entailment"])
    return dict(
      example_id=x["idx"],
      context=tf.strings.join(["it is context: ", x["premise"]]),
      question=tf.strings.join([
        "question: Does this sentence entail the following hypothesis?: ", x["hypothesis"]]),
      text_targets=_labels[x["label"]]
    )

else:
  _preprocess_rte = functools.partial(
    build_sentence_classification, sent1_key="premise", sent2_key="hypothesis",
    prompt=GLUE_PROMPTS["rte"][0], label_mapping=GLUE_PROMPTS["rte"][1])


if USE_V1_PROMPTS:

  @seqio.map_over_dataset
  def _preprocess_cb(x):
    _labels = tf.constant(['entailment', 'contradiction', 'neutral'])
    return dict(
      example_id=x["idx"],
      context=tf.strings.join(["context: ", x["premise"]]),
      question=tf.strings.join([
        "question: Does this sentence entail the following hypothesis?: ", x["hypothesis"]]),
      text_targets=_labels[x["label"]]
    )

else:
  _preprocess_cb = functools.partial(
    build_sentence_classification, sent1_key="premise", sent2_key="hypothesis",
    prompt=GLUE_PROMPTS["mnli"][0], label_mapping=['entailment', 'contradiction', 'neutral'])


add_superglue("boolq", _preprocess_boolq)
add_superglue("cb", _preprocess_cb)
add_superglue("rte", _preprocess_rte)
add_superglue("copa", _preprocess_copa)
add_superglue("multirc", _preprocess_multirc)
add_superglue("record", _preprocess_record)
add_superglue("wic", _preprocess_wic)
add_superglue("wsc.fixed", _preprocess_wsc)
add_superglue("axg", functools.partial(
  build_sentence_classification, sent1_key="premise", sent2_key="hypothesis",
  prompt=GLUE_PROMPTS["rte"][0], label_mapping=GLUE_PROMPTS["rte"][1]
))
add_superglue("axb", functools.partial(
  build_sentence_classification, sent1_key="sentence1", sent2_key="sentence2",
  prompt=GLUE_PROMPTS["rte"][0], label_mapping=GLUE_PROMPTS["rte"][1]
))

MixtureRegistry.add("super_glue", SUPER_GLUE_TASKS, default_rate=1.0)


# ==== Other Sentence Classification Dataset ====
add_sentence_classification(
  "snli", "snli:1.1.0", [exact_match], GLUE_PROMPTS["mnli"],
  "premise", "hypothesis"
)

add_sentence_classification(
  "scitail", "sci_tail:1.0.0", [exact_match], GLUE_PROMPTS["rte"],
  "premise", "hypothesis"
)

add_sentence_classification(
  "imdb_reviews", "imdb_reviews/plain_text:1.0.0", [exact_match], GLUE_PROMPTS["sst2"], "text", None)

add_sentence_classification(
  "paws_wiki", "paws_wiki/labeled_final_raw:1.1.0", [exact_match], GLUE_PROMPTS["mrpc"],
  "sentence1", "sentence2"
)
MixtureRegistry.add("sentence_classification", [
  "snli", "scitail", "imdb_reviews", "paws_wiki"
], default_rate=1.0)


# ==== SQUAD ====
def squad_metric(targets: Sequence, predictions: Sequence[str], aux_values):
  qid_to_pred = defaultdict(list)
  qid_to_answers = dict()
  for target, pred, score in zip(targets, predictions, aux_values["scores"]):
    example_id = target["example_id"].decode("utf-8").split("-")[0]
    qid_to_pred[example_id].append((pred, score))
    answers = [x.decode("utf-8") for x in target["answers"]]
    if example_id not in qid_to_answers:
      qid_to_answers[example_id] = answers
    else:
      if qid_to_answers[example_id] != answers:
        raise ValueError()

  ems, f1s = [], []
  for qid, predictions in qid_to_pred.items():
    answer = ''  # empty string is used as the text for non-answers
    best_score = None
    for pred, score in predictions:
      if pred == UNANSWERABLE_QUESTION_OUTPUT:
        continue
      if best_score is None or score > best_score:
        best_score = score
        answer = pred
    gold = qid_to_answers[qid]
    if len(gold) == 0:
      gold = ['']
    ems.append(max(squad_exact_match(g, answer) for g in gold))
    f1s.append(max(squad_f1(g, answer) for g in gold))

  return {
    "em": Scalar(np.mean(ems)),
    "f1": Scalar(np.mean(f1s))
  }


UNANSWERABLE_QUESTION_OUTPUT = tf.constant("not enough information")


@seqio.map_over_dataset
def extract_squad(x, include_title=True):
  voc = get_default_vocabulary()
  if include_title:
    header = tf.strings.join(["title: ", x["title"], "context:"])
  else:
    header = tf.constant("context:")

  # Need to add the space here not in `context:` because the T5 tokenizer might merge it
  # with the first word inx["context"]
  context, starts, ends = voc.tf_tokenizer.tokenize_with_offsets(
    tf.strings.join([" ", x["context"]]))

  return dict(
    example_id=x["id"],
    header=voc.encode_tf(header),
    context=context,
    question=voc.encode_tf(tf.strings.join(["question: ", x["question"]])),
    answer_tok=voc.encode_tf(x["answers"]["text"]),
    answers=tf.unique(x["answers"]["text"])[0],
    answer_start=x["answers"]["answer_start"]+1,
    answer_ends=x["answers"]["answer_start"]+1 - tf.strings.length(x["answers"]["text"]),
    token_bounds=tf.stack([starts, ends], -1)
  )


def sliding_window(ds, sequence_length, step_size=64, max_windows=4):

  def _build_windows(x):
    voc = get_default_vocabulary()
    question, context, header = x["question"], x["context"], x["header"]
    max_context_len = sequence_length["text_inputs"] - tf.shape(question)[0] - tf.shape(header)[0]
    extra = tf.shape(context)[0] - max_context_len
    extra_steps = tf.clip_by_value((extra+step_size-1) // step_size, 0, 50)
    steps = tf.range(0, extra_steps+1)

    unanswerable_question_tok = voc.encode_tf(UNANSWERABLE_QUESTION_OUTPUT)

    ans_start = x["answer_start"]
    ans_end = x["answer_ends"]
    bounds = tf.cast(x["token_bounds"], tf.int32)

    if max_windows is not None and extra_steps > max_windows:
      # Sub-sample the windows, can be used during training for long-document inputs
      # If at least one window contains the answer, we make one such window is included
      # in the sub-sample
      starts = steps*step_size
      ends = steps*step_size+max_context_len
      window_starts = tf.gather(bounds[:, 0], starts)
      window_ends = tf.gather(bounds[:, 1], tf.minimum(ends, tf.shape(bounds)[0])-1)

      # [n_step, n_answers]
      is_valid = tf.logical_and(
        tf.expand_dims(ans_start, 0) >= tf.expand_dims(window_starts, 1),
        tf.expand_dims(ans_end, 0) <= tf.expand_dims(window_ends, 1))
      has_valid_ans = tf.reduce_any(is_valid, -1)  # [n_step]
      if tf.reduce_any(has_valid_ans):
        ans_window = tf.where(has_valid_ans)[0]
        ans_window = ans_window[tf.random.uniform((), 0, tf.shape(ans_window)[0], dtype=tf.int32)]
        ans_window = tf.cast(ans_window, tf.int32)
        other = tf.range(0, extra_steps, dtype=tf.int32)
        other += tf.cast(other > ans_window, other.dtype)
        steps = tf.random.shuffle(other)
        steps = tf.concat([tf.expand_dims(ans_window, 0), steps[:max_windows-1]], 0)
      else:
        steps = tf.random.shuffle(steps)
        steps = steps[:max_windows]

    windows = tf.data.Dataset.from_tensor_slices(steps)

    def _build_window(i):
      start = i*step_size
      end = i*step_size+max_context_len
      text_inputs = tf.concat([header, context[start:end], question], -1)
      window_start = bounds[start, 0]
      window_end = bounds[tf.minimum(end, tf.shape(bounds)[0])-1, 1]
      is_valid = tf.logical_and(ans_start >= window_start, ans_end <= window_end)
      valid_answers = tf.ragged.boolean_mask(x["answer_tok"], is_valid)

      if tf.shape(valid_answers.values)[0] == 0:
        target_text = unanswerable_question_tok
      else:
        target_text = valid_answers[0]
      return dict(
        text_inputs=text_inputs, text_targets=target_text, answers=x["answers"],
        example_id=tf.strings.join([x["example_id"], "-window", tf.strings.as_string(i)]),
      )

    return windows.map(_build_window)

  return ds.flat_map(_build_windows)
    # .filter(lambda x: tf.shape(x["answers"])[0] > 0)\
    # .filter(lambda x: tf.shape(x["context"])[0] > 250)\


TaskRegistry.add(
  f"squad",
  source=seqio.TfdsDataSource(
    tfds_name="squad/v2.0:3.0.0",
    tfds_data_dir=TFDS_DATA_DIR
  ),
  preprocessors=[
    extract_squad,
    seqio.CacheDatasetPlaceholder(),
    functools.partial(sliding_window, max_windows=4),
    functools.partial(text_to_text_preprocessor, pass_through=("example_id", "answers"))
  ],
  output_features=FINETUNE_OUTPUT_FEATURES,
  postprocess_fn=nlp_post_processor,
  metric_fns=[squad_metric]
)


# ==== MRQA ====

@seqio.map_over_dataset
def _preprocess_mrqa(ex):
  voc = get_default_vocabulary()
  context, starts, ends = voc.tf_tokenizer.tokenize_with_offsets(
    tf.strings.join([" ", ex["context"]]))

  # The detected answers are stored as ragged tensors, so we flatten them here
  # to a get a tokenized and answer/start index/end index per a detection
  detected_answers = ex["detected_answers"]
  answer_tok = voc.encode_tf(detected_answers["text"])
  ans_starts = detected_answers["char_spans"]["start"]
  ahs_ends = detected_answers["char_spans"]["end"]
  answer_tok = tf.gather(answer_tok, ans_starts.value_rowids())

  return dict(
    example_id=ex["qid"],
    header=voc.encode_tf(tf.constant("context:")),
    context=context,
    question=voc.encode_tf(tf.strings.join(["question: ", ex["question"]])),
    answer_tok=answer_tok,
    answers=ex["answers"],
    answer_start=ans_starts.flat_values+1,
    answer_ends=ahs_ends.flat_values + 1,
    token_bounds=tf.stack([starts, ends], -1),
  )


def add_mrqa(name):
  TaskRegistry.add(
    f"mrqa___{name}",
    source=seqio.TfdsDataSource(
      tfds_name=f"mrqa/{name}:1.0.0",
      tfds_data_dir=TFDS_DATA_DIR
    ),
    preprocessors=[
      _preprocess_mrqa,
      seqio.CacheDatasetPlaceholder(),
      functools.partial(sliding_window, max_windows=4),
      functools.partial(text_to_text_preprocessor, pass_through=("example_id", "answers")),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=nlp_post_processor,
    metric_fns=[squad_metric]
  )


MRQA_TRAIN_TASKS = ["news_qa", "trivia_qa", "search_qa", "hotpot_qa", "natural_questions"]
MRQA_DEV_TASKS = ["bio_asq", "drop", "duo_rc", "race", "relation_exaction", "textbook_qa"]

for task in MRQA_TRAIN_TASKS:
  add_mrqa(task)
for task in MRQA_DEV_TASKS:
  add_mrqa(task)


MixtureRegistry.add("extractive_qa", ["mrqa___" + x for x in MRQA_TRAIN_TASKS] + ["squad"], default_rate=1.0)

MixtureRegistry.add("mrqa_val", ["mrqa___" + x for x in MRQA_DEV_TASKS], default_rate=1.0)


# ==== Multiple Choice QA ====
def add_dataset(name, tfds_name, preprocess_fn):
  TaskRegistry.add(
    name,
    source=seqio.TfdsDataSource(
      tfds_name=tfds_name,
      tfds_data_dir=TFDS_DATA_DIR,
    ),
    preprocessors=[
      preprocess_fn,
      tokenize,
      seqio.CacheDatasetPlaceholder(),
      build_text_inputs,
      functools.partial(text_to_text_preprocessor, pass_through=("example_id", "label")),
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
    postprocess_fn=nlp_post_processor,
    metric_fns=[exact_match]
  )


@seqio.map_over_dataset
def _cosmo_qa(x):
  answers = [x[f"answer{i}"] for i in range(4)]
  question = build_multiple_choice_question(x["question"], answers)
  return dict(
    example_id=x["id"],
    question=question,
    context=x["context"],
    text_targets=tf.stack(answers, 0)[x["label"]]
  )


@seqio.map_over_dataset
def _hella_swag(x):
  answers = x["endings"]
  question = build_multiple_choice_question("Which ending is the most likely?", answers)
  return dict(
    example_id=x["source_id"],
    question=question,
    context=x["context"],
    text_targets=tf.stack(answers, 0)[x["label"]]
  )


@seqio.map_over_dataset
def _piqa(x):
  choices = [x["sol1"], x["sol2"]]
  question = build_multiple_choice_question("Which solution is best?", choices)
  return dict(
    example_id=x["id"],
    question=question,
    context=tf.strings.join(["context: ", x["goal"]]),
    text_targets=tf.stack(choices)[x["label"]]
  )


def _openbook_qa(ds):
  def _map(i, x):
    answers = [x["question"][f"choice_{key}"] for key in ["A", "B", "C", "D"]]
    question = build_multiple_choice_question(x["question"]["stem"], answers)
    return dict(
      example_id=i,
      question=question,
      context=x["fact1"],
      text_targets=tf.stack(answers, 0)[x["answerKey"]]
    )
  return ds.enumerate().map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


add_dataset("piqa", "piqa:1.0.0", _piqa)
add_dataset("cosmos_qa", "cosmos_qa:1.0.0", _cosmo_qa)
add_dataset("hellaswag", "hellaswag:1.1.0", _hella_swag)
add_dataset("openbookqa", "openbookqa:0.1.0", _openbook_qa)

MixtureRegistry.add("mc_qa", ["piqa", "cosmos_qa", "hellaswag", "openbookqa"], default_rate=1.0)


# ==== Summerization ===

@seqio.map_over_dataset
def _gigaword(x):
  return dict(
    context=tf.strings.join(["context: ", x["document"]]),
    question=tf.constant("question: What is a one sentence summary of this document?"),
    text_targets=x["summary"]
  )


TaskRegistry.add(
  "gigaword",
  source=seqio.TfdsDataSource(
    tfds_name="gigaword:1.2.0",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    _gigaword,
    tokenize,
    seqio.CacheDatasetPlaceholder(),
    build_text_inputs,
    text_to_text_preprocessor
  ],
  output_features=FINETUNE_OUTPUT_FEATURES,
)


MixtureRegistry.add(
  "nlp",
  [
    "gigaword", "glue", "super_glue", "mc_qa", "extractive_qa", "sentence_classification"
  ],
  default_rate=1.0
)