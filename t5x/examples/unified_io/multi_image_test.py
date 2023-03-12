import os

import gin
import jax
import seqio

from t5x import optimizers
from t5x.examples.unified_io.data import cmumosei  # noqa
from t5x.examples.unified_io.data import visual_dialogue_meta_tasks  # noqa
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary, UnifiedIOFeatureConverter
from t5x.examples.unified_io.multi_image_models import MultiImageEncoderDecoderModel
from t5x.examples.unified_io.multi_image_network import VideoTransformer
from t5x.examples.unified_io.network import T5Config, VAEConfig

if __name__ == "__main__":
  gin_bindings = [
    'DROPOUT_RATE = 0.0',
    'TEXT_DECODER_LENGTH = 128',
    'IMAGE_DECODER_LENGTH = 1',
  ]
  abs_path = os.path.dirname(os.path.abspath(__file__))
  gin_files = [os.path.join(abs_path, './t5_1_1/multi_large.gin')]

  with gin.unlock_config():
    gin.parse_config_files_and_bindings(
      gin_files, gin_bindings, finalize_config=False)
  config = gin.get_configurable(T5Config)()
  vae_config = gin.get_configurable(VAEConfig)()
  vocabulary = get_default_vocabulary()

  module = VideoTransformer(config=config, vae_config=vae_config)
  model = MultiImageEncoderDecoderModel(module=module,
                                        input_vocabulary=vocabulary,
                                        output_vocabulary=vocabulary,
                                        optimizer_def=optimizers.sgd(0.1))

  # datasets
  # dataset = seqio.get_mixture_or_task("ytdialogue:1.5.0").get_dataset(
  #   sequence_length={"text_inputs": 256,
  #                    "text_targets": 128,
  #                    "image_input_samples": 576,
  #                    "is_training": True},
  #   split="test",
  #   shuffle=False,
  # )

  dataset = seqio.get_mixture_or_task("cmumosei:sentiment:1.0.0").get_dataset(
    sequence_length={"text_inputs": 256,
                     "text_targets": 128,
                     "image_input_samples": 576,
                     "is_training": True},
    split="train",
    shuffle=False,
  )
  converter = UnifiedIOFeatureConverter(pack=False, use_custom_packing_ops=False)
  dataset = converter(dataset, {"text_inputs": 256, "text_targets": 128})
  for raw_record in dataset.take(1):
    rng = jax.random.PRNGKey(0)
    input_shapes = {k: [1] + v.shape for k, v in raw_record.items()}
    print(input_shapes)
    variables = model.get_initial_variables(rng, input_shapes)
    import ipdb;
    ipdb.set_trace();
