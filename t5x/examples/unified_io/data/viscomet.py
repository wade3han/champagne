import functools

import seqio
from seqio import TaskRegistry, MixtureRegistry

from t5x.examples.unified_io.data.metrics import coco_captioning_metric, save_predictions_vqa
from t5x.examples.unified_io.data.postprocessors import viscomet_postprocessor, get_id
from t5x.examples.unified_io.data.preprocessors import rekey, viscommet_preprocessor
from t5x.examples.unified_io.data.task_constants import FINETUNE_OUTPUT_FEATURES, FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES
from t5x.examples.unified_io.data.task_utils import TFDS_DATA_DIR

TaskRegistry.add(
  "vis_commet_before_single",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      mode='before'
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_after_single",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      mode='after'
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_intent_single",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      mode='intent'

    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_before_multi",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='before'
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_after_multi",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='after'
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_intent_multi",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
    splits={"train": "train", "validation": "validation", "test": "validation"},
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "before": ["before"],
        "after": ["after"],
        "intent": ["intent"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "intent_bbox_id": ["intent_bbox_id"],
        "before_bbox_id": ["before_bbox_id"],
        "after_bbox_id": ["after_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='intent'

    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=viscomet_postprocessor,
  metric_fns=[coco_captioning_metric],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)

# 111,796 samples for each, total 335,388 samples
MixtureRegistry.add('vis_commet_multi',
                    ['vis_commet_intent_multi', 'vis_commet_before_multi', 'vis_commet_after_multi'],
                    default_rate=1.0)

MixtureRegistry.add('vis_commet_single',
                    ['vis_commet_intent_single', 'vis_commet_before_single', 'vis_commet_after_single'],
                    default_rate=1.0)


TaskRegistry.add(
  "vis_commet_before_multi_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='before',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='before')],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)


TaskRegistry.add(
  "vis_commet_after_multi_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='after',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='after')],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)


TaskRegistry.add(
  "vis_commet_intent_multi_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=True,
      mode='intent',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='intent')],
  output_features=FINETUNE_MULTI_IMAGE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_before_single_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=False,
      mode='before',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='before')],
  output_features=FINETUNE_OUTPUT_FEATURES,
)


TaskRegistry.add(
  "vis_commet_after_single_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=False,
      mode='after',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='after')],
  output_features=FINETUNE_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vis_commet_intent_single_test",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="vis_commet_test_official:1.0.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "place": ["place"],
        "event": ["event"],
        "id": ["id"],
        "place_bbox_id": ["place_bbox_id"],
        "event_bbox_id": ["event_bbox_id"],
        "names": ["names"],
        "boxes": ["boxes"],
      }),
    functools.partial(
      viscommet_preprocessor,
      use_multiple_images=False,
      mode='intent',
      eval=True,
    ),
    seqio.preprocessors.append_eos_after_trim,
  ],
  postprocess_fn=get_id,
  metric_fns=[
    functools.partial(save_predictions_vqa, task='intent')],
  output_features=FINETUNE_OUTPUT_FEATURES,
)


# 111,796 samples for each, total 335,388 samples
MixtureRegistry.add('vis_commet_multi_test',
                    ['vis_commet_intent_multi_test', 'vis_commet_before_multi_test', 'vis_commet_after_multi_test'],
                    default_rate=1.0)

MixtureRegistry.add('vis_commet_single_test',
                    ['vis_commet_intent_single_test', 'vis_commet_before_single_test', 'vis_commet_after_single_test'],
                    default_rate=1.0)
