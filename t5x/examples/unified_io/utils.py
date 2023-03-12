import logging
import os
import re
import tempfile
from dataclasses import dataclass
from os.path import join, exists
from typing import Optional, Any, List

import gin
import numpy as np
import requests
from tensorflow.io import gfile
from tqdm import tqdm

from t5x import utils


@gin.configurable()
def init_wandb(name=None, group=None, entity=None, project=None):
  utils.create_learning_rate_scheduler()  # Make sure this is registered in `operative_config`
  config_str = gin.operative_config_str()
  print(config_str)

  # This is a bit stupid, but parsing this string seems like the easiest way to get
  # the value as a dictionary
  config_vals = {}
  for match in re.compile("([A-z0-9_\.]+) = (.+)").finditer(config_str):
    val = match.group(2)
    if val[0] == '\'' and val[-1] == '\'':
      val = val[1:-1]
    else:
      try:
        val = int(val)
      except ValueError:
        try:
          val = float(val)
        except ValueError:
          pass
    config_vals[match.group(1)] = val
  wandb_config = dict()

  # Pick parameter to store in wandb, I picked out some parameters that I think might be changed
  # for now. I am storing them in with the gin.config name just to keep a 1-to-1 mapping
  for k in [
    "BATCH_SIZE",
    "DROPOUT_RATE",
    'MIXTURE_OR_TASK_NAME',
    "TRAIN_STEPS",
    "Z_LOSS",
    'LABEL_SMOOTHING',
    # "INITIAL_CHECKPOINT_PATH",
    'network.T5Config.default_image_size',
    "network.T5Config.decoder_max_image_length",
    "network.T5Config.decoder_max_text_length",
    'network.T5Config.encoder_max_image_length',
    'network.T5Config.encoder_max_text_length',
    'network.T5Config.image_vocab_size',
    'train_script.train.eval_period',
    'utils.create_learning_rate_scheduler.base_learning_rate',
    'utils.create_learning_rate_scheduler.factors',
    'utils.create_learning_rate_scheduler.warmup_steps',

  ]:
    wandb_config[k] = config_vals[k]

  import wandb
  logging.info(f"Init wandb with group={group} name={name}, entity={entity}, project={project}")
  wandb.init(
    group=group,
    name=name,
    entity=entity,
    project=project,
    force=True,
    config=wandb_config,
    notes=gin.operative_config_str()
  )


def download(ckpt_dir, url):
  name = url[url.rfind('/') + 1 : url.rfind('?')]
  if ckpt_dir is None:
    ckpt_dir = tempfile.gettempdir()
  ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
  ckpt_file = os.path.join(ckpt_dir, name)
  if not os.path.exists(ckpt_file):
    print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    # first create temp file, in case the download fails
    ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
    with open(ckpt_file_temp, 'wb') as file:
      for data in response.iter_content(chunk_size=1024):
        progress_bar.update(len(data))
        file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
      print('An error occured while downloading, please try again.')
      if os.path.exists(ckpt_file_temp):
        os.remove(ckpt_file_temp)
    else:
      # if download was successful, rename the temp file
      os.rename(ckpt_file_temp, ckpt_file)
  return ckpt_file



def html_rect(x1, y1, x2, y2, rel=None, rank=None, color="black", border_width="medium", label=None):
  rect_style = {
    "position": "absolute",
    "top": y1,
    "left": x1,
    "height": y2-y1,
    "width": x2-x1,
    "border-style": "solid",
    "border-color": color,
    "border-width": border_width,
    "box-sizing": "border-box"
  }
  rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

  text_style = {
    "position": "absolute",
    "top": y1-5,
    "left": x1+3,
    "color": color,
    "background-color": "black",
    "z-index": 9999,
    "padding-right": "5px",
    "padding-left": "5px",
  }
  text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

  if rel is None and rank is None:
    container = ""
  else:
    container = f'class=box'
    if rel:
      container += f' data-rel="{rel}"'
    if rank:
      container += f' data-rank="{rank}"'

  if rel is None and label is None:
    text = ''
  elif label is not None:  # Label overrides rel
    text = f'  <div style="{text_style_str}">{label}</div>'
  else:
    text = f'  <div style="{text_style_str}">{rel:0.2f}</div>'

  html = [
    f'<div {container}>',
    f'  <div style="{rect_style_str}"></div>',
    text,
    "</div>"
  ]
  return html


def load_gs_file_with_cache(src, cache) -> str:
  assert src.startswith("gs://")
  if cache is not None:
    cache_file = src[len("gs://"):]
    cache_file = cache_file.replace("/", "-")
    cache_file = join(cache, cache_file)
    if exists(cache_file):
      with open(cache_file) as f:
        return f.read()

  if not gfile.exists(src):
    raise ValueError(f"File {src} not found!")
  with gfile.GFile(src) as f:
    data = f.read()
  if cache is not None:
    with open(cache_file, "w") as f:
      f.write(data)
  return data


@dataclass
class BoxesToVisualize:
  boxes: Any
  format: str
  color: str
  scores: Optional[np.ndarray]=None
  normalized: bool = False
  labels: List[str] = None


def get_image_html_boxes(image_src, boxes: List[BoxesToVisualize],
                         width=None, height=None, wrap="div", img_size=None):
  html = []
  html += [f'<{wrap} style="display: inline-block; position: relative;">']

  image_attr = dict(src=image_src)
  if width:
    image_attr["width"] = width
  if height:
    image_attr["height"] = height
  attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
  html += [f'<img {attr_str}>']

  for box_set in boxes:
    if box_set.normalized:
      raise NotImplementedError()
    else:
      if height or width:
        if img_size is None:
          with PIL.Image.open(image_src) as img:
            img_w, img_h = img.size[:2]
        else:
          img_w, img_h = img_size
        if not width:
          factor = height/img_h
          w_factor = factor
          h_factor = factor
        else:
          raise NotImplementedError()
      else:
        w_factor = 1
        h_factor = 1

    task_rel = box_set.scores
    task_boxes = box_set.boxes
    if task_boxes is not None and len(task_boxes) > 0:
      if box_set.format == "yxyx":
        task_boxes = np.stack([
          task_boxes[:, 1], task_boxes[:, 0],
          task_boxes[:, 3], task_boxes[:, 2],
        ], -1)
      elif box_set.format == "xyxy":
        pass
      elif box_set.format == "xywh":
        task_boxes = np.stack([
          task_boxes[:, 0], task_boxes[:, 1],
          task_boxes[:, 0] + task_boxes[:, 2],
          task_boxes[:, 1] + task_boxes[:, 3]
        ], -1)
      else:
        raise NotImplementedError(box_set.format)

    if task_rel is not None:
      ixs = np.argsort(-task_rel)
    else:
      ixs = np.arange(len(task_boxes))
    for rank, ix in enumerate(ixs):
      box = task_boxes[ix]
      rel = None if task_rel is None else task_rel[ix]
      x1, y1, x2, y2 = box
      html += html_rect(
        x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor,
        rel=rel, rank=rank+1,
        color=box_set.color,
        label=None if box_set.labels is None else box_set.labels[ix]
      )

  html += [f'</{wrap}>']
  return html
