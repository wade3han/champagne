import functools

import seqio
import tensorflow as tf
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.evaluation import metrics
import tensorflow_datasets as tfds

TaskRegistry = seqio.TaskRegistry


DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


TaskRegistry.add(
    "c4_v220_span_corruption",
    source=seqio.TfdsDataSource(
        tfds_name="c4/en:2.3.1",
        tfds_data_dir='gs://armanc-tpu-bucket/tensorflow_datasets',
        ),
        preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])