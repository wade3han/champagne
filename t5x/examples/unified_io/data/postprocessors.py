import re

from t5x.examples.unified_io.data.data_utils import get_default_vocabulary

vocab = get_default_vocabulary()


def lower_text(inputs, **unused_kwargs):
  """Lowercases text."""
  text, image = inputs
  return text.lower()


def refexp_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return [example["box"], example["box_size"]]
  else:
    return output_or_target


def basic_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return example
  else:
    return output_or_target


def pose_postprcoessing(output_or_target, example=None, is_target=False):
  if is_target:
    return example
  else:
    return output_or_target


def image_caption_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return [a.decode("utf-8").lower() for a in example["all_references"]]
  else:
    return output_or_target


def viscomet_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return [re.sub(r'<extra_id_(.*?)> ', r'', a.decode("utf-8").lower()) for a in example["all_references"]]
  else:
    if isinstance(output_or_target, str):
      output_or_target = re.sub(r'<extra_id_(.*?)> ', r'', output_or_target)
    else:
      output_or_target.text = re.sub(r'<extra_id_(.*?)> ', r'', output_or_target.text)
    return output_or_target


def image_tagging_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    raise ValueError()
  else:
    return output_or_target


def get_id(output_or_target, example=None, is_target=False):
  if is_target:
    return example["example_id"], example["image_info"]
  else:
    return output_or_target


def vqa_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return [a.decode("utf-8").lower() for a in example["all_references"]]
  else:
    return output_or_target


def visdial_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return example["image_id"], example["round_id"], \
           example['text_inputs_pretokenized'].decode('utf-8'), example['text_targets_pretokenized'].decode('utf-8')
  else:
    return output_or_target


def dialogue_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return example['text_targets_pretokenized'].decode('utf-8')
  else:
    return output_or_target


def dialogue_postprocessor_infer(output_or_target, example=None, is_target=False):
  if is_target:
    return example['text_targets_pretokenized'].decode('utf-8'), example['text_inputs_pretokenized'].decode('utf-8')
  else:
    return output_or_target


def vcr_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return example["annot_ids"].decode('utf-8'), \
           example["answer_label"], \
           example["rationale_label"], \
           vocab.decode(example["text_targets"]), \
           vocab.decode(example["text_inputs"])
  else:
    return output_or_target


def tvc_postprocessor(output_or_target, example=None, is_target=False):
  if is_target:
    return example["clip_id"][0].decode('utf-8')
  else:
    return output_or_target


def extract_text_label(output_or_target, example=None, is_target=False):
  if is_target:
    return example["label"].decode("utf-8")
  else:
    return output_or_target


def extract_depth(output_or_target, example=None, is_target=False):
  if is_target:
    return {k: example[k] for k in ["depth", "image_info"]}
  else:
    return output_or_target


def return_example(output_or_target, example=None, is_target=False):
  if is_target:
    return example
  else:
    return output_or_target


def extract_class_segmentation(output_or_target, example=None, is_target=False):
  if is_target:
    return {k: example[k] for k in ["image_targets", "image_info"]}
  else:
    return output_or_target


def framenet_postprocess(output_or_target, example=None, is_target=False):
  if is_target:
    return {k: example[k] for k in ["gt_image", "image_info"]}
  else:
    return output_or_target


def framenet_flat_postprocess(output_or_target, example=None, is_target=False):
  if is_target:
    return {k: example[k] for k in ["gt_image", "image_info", "example_id", "axis"]}
  else:
    return output_or_target


def extract_panoptic_segmentation(output_or_target, example=None, is_target=False):
  if is_target:
    return example
  else:
    return output_or_target


def extract_bboxes(output_or_target, example=None, is_target=False):
  if is_target:
    return example["boxes"], example["labels"], example["image_info"], example.get("src_boxes")
  else:
    return output_or_target
