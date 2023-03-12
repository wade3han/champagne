import dataclasses
import json
from typing import Dict, List, Optional, Mapping, Sequence, Any

import gin
import tensorflow_datasets as tfds
from seqio import metrics as metrics_lib
import itertools
import seqio
from seqio.evaluation import AllMetricsType

from t5x import partitioning
import tensorflow as tf
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from t5x.utils import InferStepCallable, PartitionSpec
from absl import logging
from t5x import train_state as train_state_lib

AllOutputTokensType = Mapping[str, Sequence[Sequence[int]]]
AllOutputScoresType = Mapping[str, Sequence[float]]
AllOutputAuxValuesType = Mapping[str, Mapping[str, Sequence[Any]]]
AllMetricsType = Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass
class UnifiedIOOutput:
  text: str
  scores: float
  image: Optional[np.ndarray] = None
  logprobs: np.array = None
  text_tokens: Optional[np.ndarray] = None

  def to_dict(self):
    return dict(self.__dict__)


def build_uio_outputs(predictions, aux_values) -> List[UnifiedIOOutput]:
  out = []
  for ix in range(len(predictions)):
    out.append(UnifiedIOOutput(
      predictions[ix], aux_values["scores"][ix],
      image=None if "img" not in aux_values else aux_values["img"][ix],
      text_tokens=None if "txt_tokens" not in aux_values else aux_values["txt_tokens"][ix]
    ))
  return out


@gin.configurable()
class UnifiedIOEvaluator(seqio.Evaluator):
  log_wandb: bool = False
  def _compute_metrics(self,
                       predicted_tokens: AllOutputTokensType,
                       scores: AllOutputScoresType,
                       all_aux_values: AllOutputAuxValuesType,
                       step: Optional[int] = None) -> AllMetricsType:
    """Computes and logs metrics given the predicted tokens and scores.
    Args:
      predicted_tokens: a mapping from task name to the output tokens from
        `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
       , `score_fn` for tasks that have `score_predict_fns`.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
    Returns:
      A mapping from task name to computed metrics.
    """
    # Later version of Seqio have this method take a
    # `all_aux_values` flag, which we don't use anway
    # so check for it here and ignore it
    all_metrics = {}

    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]
      targets = self.cached_targets[task.name]
      task_metrics = []
      inferences = {}

      if task.predict_metric_fns or task.predict_with_aux_metric_fns:
        (outputs,
         postprocessed_outputs) = self._decode_and_postprocess_predictions(
             task, predicted_tokens, task_dataset, targets)
        inferences["output"] = outputs
        inferences["prediction"] = postprocessed_outputs

      if task.predict_metric_fns:
        task_metrics.extend([
            metric_fn(targets, inferences["prediction"])
            for metric_fn in task.predict_metric_fns
        ])

      if task.predict_with_aux_metric_fns:
        aux_values = all_aux_values[task.name]
        uio_output = build_uio_outputs(inferences["prediction"], aux_values)
        task_vocab = task.output_features["text_inputs"].vocabulary
        text_inputs = [ex["text_inputs"] for ex in tfds.as_numpy(task_dataset)]
        text_inputs = [task_vocab.decode(text_input) for text_input in text_inputs]
        aux_values["text_inputs"] = text_inputs
        aux_inputs = text_inputs
        if "ytdialogue_denoise" in task.name or "yttemporal" in task.name:
          aux_values["youtube_ids"] = [ex["youtube_id"] for ex in tfds.as_numpy(task_dataset)]
          aux_values["start_secs"] = [ex["start_sec"] for ex in tfds.as_numpy(task_dataset)]
          aux_values["end_secs"] = [ex["end_sec"] for ex in tfds.as_numpy(task_dataset)]

        task_metrics.extend([
            metric_fn(targets, uio_output, aux_inputs)
            for metric_fn in task.predict_with_aux_metric_fns
        ])
        inferences["aux_value"] = aux_values

      if task.score_metric_fns:
        logging.info("Computing score metrics.")
        task_scores = scores[task.name]
        # if task.name == "visdial:ndcg" or \
        #     task.name == "visdial:val_official" or \
        #     task.name == "visdial:test" or \
        #     task.name == "visdial:1.1.0:ndcg" or \
        #     task.name == "visdial:1.1.0:val_official" or \
        #     task.name == "visdial:1.1.0:val_official:pmi" or \
        #     task.name == "visdial:1.1.0:test" or \
        #     task.name == "visdial:1.1.0:test:pmi" or \
        #     task.name == "visdial:1.2.0:ndcg" or \
        #     task.name == "visdial:1.2.0:val_official" or \
        #     task.name == "visdial:1.2.0:val_official:pmi" or \
        #     task.name == "visdial:1.2.0:test" or \
        #     task.name == "visdial:1.2.0:test:pmi":
        #   logging.info("Visdial ---")
        #   for ex, target in zip(tfds.as_numpy(task_dataset), targets):
        #     metric_targets.append((ex['image_id'],
        #                            ex['round_id'],
        #                            task_vocab.decode(ex['text_inputs']),
        #                            target))
        # elif task.name in ["eval_vcr_qa", "eval_vcr_qar"]:
        #   ...
        # else:
        #   for ex, target in zip(tfds.as_numpy(task_dataset), targets):
        #     metric_targets.append((task_vocab.decode(ex['text_inputs']),
        #                            target))
        if len(targets) != len(task_scores):
          raise ValueError(f"len(targets)({len(targets)}) != "
                           f"len(task_scores)({len(task_scores)})")
        task_metrics.extend([
            metric_fn(targets, task_scores)
            for metric_fn in task.score_metric_fns
        ])
        inferences["score"] = task_scores

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      metrics = {
        k: metrics_lib.Scalar(v)
        if not isinstance(v, metrics_lib.MetricValue) else v
        for k, v in all_metrics[task.name].items()
      }
      for logger in self.loggers:
        logger(task_name=task.name, step=step, metrics=metrics,
               dataset=task_dataset, inferences=inferences, targets=targets)
      for task_name in all_metrics:
        metric = all_metrics[task_name]
        metric.pop('dummy', None)  # too noisy to print
    return all_metrics
