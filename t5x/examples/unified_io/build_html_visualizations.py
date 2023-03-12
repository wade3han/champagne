import base64
import io
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from typing import Any, Dict, Union, List

import numpy as np
import torch

from t5x.examples.unified_io.data.data_utils import unnormalize_image
from t5x.examples.unified_io.data.metrics import build_depth_prediction, depth_example_metrics
from t5x.examples.unified_io.evaluator import UnifiedIOOutput
import tensorflow as tf


ImageData = Union[np.ndarray, torch.Tensor, tf.Tensor]
# f = open("predicted.jpg", "wb"); f.write(tf.image.encode_jpeg(tf.image.convert_image_dtype(dec, tf.uint8)).numpy()); f.close()


def build_embedding_image(image_data: ImageData, channel_first=False, fmt="jpeg"):
  """Turns an image as a string that can be used src in html images"""
  if not isinstance(image_data, torch.Tensor):
    image_data = tf.convert_to_tensor(image_data)

  assert fmt == "jpeg"
  assert not channel_first

  if image_data.dtype == tf.bool:
    # Boolean mask, convert to black/white
    if len(image_data.shape) == 2:
      image_data = tf.expand_dims(image_data, -1)
    else:
      assert image_data.shape[2] == 1
    black_constant = tf.reshape(tf.constant([0, 0, 0], dtype=tf.uint8), [1, 1, 3])
    white_constant = tf.reshape(tf.constant([255, 255, 255], dtype=tf.uint8), [1, 1, 3])
    image_data = tf.cast(image_data, tf.uint8)
    image_data = image_data*white_constant + (1-image_data)*black_constant

  elif image_data.dtype != tf.uint8:
    image_data = tf.image.convert_image_dtype(image_data, tf.uint8)

  data = tf.image.encode_jpeg(image_data, name="save_me")
  data = data.numpy()
  encoded_image = base64.b64encode(data)
  return f'data:image/{fmt};base64,{encoded_image.decode()}'


def build_html_table(data: List[Dict[str, Union[str, ImageData]]]):
    columns = {}  # Collect any key that appears in the data, in order
    for row in data:
      for key in row:
        columns[key] = None

    html = []
    html.append("<table>")

    # Header
    html.append("<tr>")
    html.append(" ".join(f"<th>{c}</th>" for c in columns))
    html.append("</tr>")

    # Body
    for ex in data:
      cells = []
      for c in columns:
        val = ex.get(c)
        if val is None:
          cells.append("")
        elif isinstance(val, (float, int, str)):
          cells.append(val)
        else:
          data = build_embedding_image(val, fmt="jpeg")
          cells.append(f'<img src={data}></img>')
      html.append("<tr>")
      html.append("\n".join(f"<td>{x}</td>" for x in cells))
      html.append("</tr>")

    html.append("</table>")
    return "\n".join(html)


def build_html_table_from_dict(stats, fmt="0.3f"):
  html = ['<table>']
  for k, v in stats.items():
    html.append(f"<tr><td>{k}</td>")
    if isinstance(v, int):
      html.append(f"<td>{v}</td></tr>")
    else:
      html.append(f"<td>{v:{fmt}}</td></tr>")
  html += ["</table>"]
  return "\n".join(html)


def build_depth_table_from_batches(batch_predictions, batches, max_depth, n_to_show=None):
  viz = []
  all_scores = []
  for (_, aux), batch in zip(batch_predictions, batches):
    for i in range(len(aux['img'])):
      pred = build_depth_prediction(aux["img"][i], batch["image_info"][i], max_depth)
      gt_depth = batch["depth"][i]
      scores = depth_example_metrics(gt_depth, pred, 0.001, 10.0)
      all_scores.append(scores)
      errors = np.clip(np.abs(gt_depth - pred)/10.0, 0, 1)
      errors_small = np.clip(np.abs(gt_depth - pred), 0, 3) / 3.0
      invalid = np.logical_not(np.logical_and(gt_depth > 0.001, gt_depth < max_depth))
      errors[invalid] = 0.0
      errors_small[invalid] = 0.0
      viz.append(dict(
        # input_image=unnormalize_image(batch["image_encoder_inputs"][i]),
        number=len(viz) + 1,
        input_image=batch["image"][i]/255.0,
        target_image=gt_depth / max_depth,
        predicted_depth=pred / max_depth,
        scores=build_html_table_from_dict(depth_example_metrics(gt_depth, pred, 0.001, 10.0)),
        errors=errors,
        errors_small=errors_small,
        # predicted_image=np.clip((aux["img"][i] + 1)/2.0, 0, 1),
      ))

  mean_scores = defaultdict(float)
  for result in all_scores:
    for k, v in result.items():
      mean_scores[k] += v
  mean_scores = {k: v/len(all_scores) for k, v in mean_scores.items()}
  mean_scores["n_examples"] = len(all_scores)

  if n_to_show is not None:
    viz = viz[:n_to_show]

  return "<h3>Metrics</h3>\n" + build_html_table_from_dict(mean_scores) + "\n<h3>Example</h3>\n" + build_html_table(viz)
