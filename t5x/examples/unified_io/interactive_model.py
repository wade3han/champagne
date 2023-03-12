import functools
import os.path

import gin
import jax
import numpy as np
import tensorflow as tf

import t5x
from t5x.examples.unified_io import decoding
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from t5x.examples.unified_io.data.preprocessors import video_to_text_preprocessor, text_to_text_preprocessor
from t5x.examples.unified_io.data.task_constants import FINETUNE_OUTPUT_FEATURES
from t5x.examples.unified_io.frozen_champagne import FrozenVideoTransformerV2
from t5x.examples.unified_io.multi_image_models import MultiImageEncoderDecoderModel
from t5x.examples.unified_io.multi_image_network import VideoTransformer
from t5x.examples.unified_io.network import T5Config, VAEConfig


class InteractiveModel:

  def __init__(self, checkpoint_path, ngram_block_size, context_ngram_block_size, min_beam_length, decode_fn):
    if "base" in checkpoint_path:
      model_type = "base"
    elif "large" in checkpoint_path:
      model_type = "large"
    elif "frozen" in checkpoint_path:
      model_type = "frozen"
    elif "xl" in checkpoint_path:
      model_type = "xl"
    else:
      raise NotImplementedError

    self.decode_fn_name = decode_fn
    if decode_fn == "beam":
      self.decode_fn = functools.partial(decoding.beam_search,
                                         ngram_block_size=ngram_block_size,
                                         context_ngram_block_size=context_ngram_block_size,
                                         min_beam_length=min_beam_length)
    elif decode_fn == "temperature":
      self.decode_fn = functools.partial(decoding.temperature_sample,
                                         temperature=0.3,
                                         topk=5,
                                         topp=0.0,
                                         ngram_block_size=ngram_block_size,
                                         context_ngram_block_size=context_ngram_block_size,
                                         min_beam_length=min_beam_length)
    else:
      raise NotImplementedError

    # setup
    self.setup(model_type)
    self.model_type = model_type

    # Build Vocabularies.
    self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=4)
    self.vocabulary = get_default_vocabulary()

    self.batch_size = 2

    # Create a T5X model.
    self.model = self._load_model(model_type)

    self.restore_from_checkpoint(checkpoint_path)


  def restore_from_checkpoint(self, checkpoint_path):
    """Restore training state from checkpoint, resets self._predict_fn()."""
    train_state_initializer = t5x.utils.TrainStateInitializer(
      optimizer_def=None,
      init_fn=self.model.get_initial_variables,
      input_shapes=self.input_shapes,
      partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
      path=checkpoint_path, mode='specific', dtype='float32', strict=False)

    train_state_axes = train_state_initializer.train_state_axes
    self._train_state = train_state_initializer.from_checkpoint_or_scratch(
      [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))
    self._predict_fn = self._get_predict_fn(train_state_axes)

  def _get_predict_fn(self, train_state_axes):
    """Generate a partitioned prediction function for decoding."""

    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_interactive(
        params, batch,
        self.decode_fn,
        self.decode_fn_name,
        num_decodes=5,
        return_all_decodes=False,
        text_length=self.text_targets,
        image_length=1,
        decoder_params={'decode_rng': None})

    return self.partitioner.partition(
      partial_predict_fn,
      in_axis_resources=(
        train_state_axes.params,
        t5x.partitioning.PartitionSpec('data', ), None),
      out_axis_resources=t5x.partitioning.PartitionSpec('data', )
    )

  def predict_tokens(self, batch, seed=0):
    """Predict tokens from preprocessed dataset batch."""
    predictions, scores = self._predict_fn(
      self._train_state.params, batch, jax.random.PRNGKey(seed))
    return predictions

  def setup(self, model_type):
    self.text_inputs = 128
    self.text_targets = 32
    self.image_input_samples = 576
    self.image_dim = 768
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if model_type == "base":
      gin_files = [os.path.join(abs_path, './t5_1_1/multi_base.gin')]
    elif model_type == "large":
      gin_files = [os.path.join(abs_path, './t5_1_1/multi_large.gin')]
    elif model_type == "xl":
      gin_files = [os.path.join(abs_path, './t5_1_1/multi_xl.gin')]
    elif model_type == "frozen":
      gin_files = [os.path.join(abs_path, './t5_1_1/frozen_xl_v2.gin')]
    else:
      raise NotImplementedError

    self.task_feature_lengths = {'text_inputs': self.text_inputs,
                                 'text_targets': self.text_targets,
                                 'image_input_samples': self.image_input_samples,
                                 'is_training': False
                                 }
    self._parse_gin(gin_files)

  @property
  def input_shapes(self):
    return {
      "text_encoder_inputs": (self.batch_size, self.text_inputs),
      "text_decoder_inputs": (self.batch_size, self.text_targets),
      "image_encoder_inputs": (self.batch_size, 1, self.image_input_samples, self.image_dim),
      "image_decoder_targets": (self.batch_size, 256, 256, 3),
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
      'DROPOUT_RATE = 0.0',
      'TEXT_DECODER_LENGTH = 128',
      'IMAGE_DECODER_LENGTH = 1',
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
        gin_files, gin_bindings, finalize_config=False)

  def _load_model(self, model_type):
    """Load up a T5X `Model` after parsing training gin config."""
    config = gin.get_configurable(T5Config)()
    vae_config = gin.get_configurable(VAEConfig)()
    if model_type == "frozen":
      _module = FrozenVideoTransformerV2(config=config, vae_config=vae_config)
    else:
      _module = VideoTransformer(config=config, vae_config=vae_config)
    return MultiImageEncoderDecoderModel(
      module=_module,
      input_vocabulary=self.vocabulary,
      output_vocabulary=self.vocabulary,
      decode_fn=self.decode_fn,
      optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
      text_decoder_length=128,
      image_decoder_length=1,
    )

  def build_dataset(self, dialogue_input, images):
    prompt = dialogue_input.split("\t")[0]
    dialogue_input = dialogue_input.split("\t")[1]
    dialogue_turns = [turn.strip() for turn in dialogue_input.split("<sep>")]
    if self.model_type == 'frozen':
      context = prompt + ' <sep> ' + ' <turn> '.join(dialogue_turns)
      text_inputs = self.vocabulary.encode_tf(context)
    else:
      turn_tokens = tf.tile(tf.reshape(self.vocabulary.encode_tf('<extra_id_0>'),
                                       [1, 1]), [len(dialogue_turns), 1])
      context = tf.concat([turn_tokens, self.vocabulary.encode_tf(dialogue_turns)], axis=1)
      prompt_tokens = tf.concat([self.vocabulary.encode_tf('<extra_id_1>'), self.vocabulary.encode_tf(prompt)], axis=0)
      text_inputs = tf.concat([prompt_tokens, context.values], axis=0)
    if len(images) == 1:
      image = np.tile(np.expand_dims(images[0], 0), (3, 1, 1, 1))
    else:
      image = np.stack(images)
    ds = tf.data.Dataset.from_tensors({
      'image': image,
      'text_inputs': text_inputs,
      'text_targets': self.vocabulary.encode_tf(' '),
    }).repeat(self.batch_size)
    return ds

  def __call__(self, images, dialogue_input, use_image: bool = True, debug: bool = False):
    if debug:
      print("Building dataset...")
    ds = self.build_dataset(dialogue_input, images)
    if debug:
      print("==> Done")
    if use_image:
      ds = video_to_text_preprocessor(ds,
                                      decode_jpeg=False,
                                      use_multiple_images=True,
                                      output_features=FINETUNE_OUTPUT_FEATURES,
                                      sequence_length=self.task_feature_lengths)
    else:
      ds = text_to_text_preprocessor(ds,
                                     use_multiple_images=True,
                                     output_features=FINETUNE_OUTPUT_FEATURES,
                                     sequence_length=self.task_feature_lengths)
    if debug:
      print("==> Preprocessing done")
    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
      ds, task_feature_lengths=self.task_feature_lengths)
    model_ds = model_ds.batch(self.batch_size)

    inferences = (results for batch in model_ds.as_numpy_iterator()
                  for results in self.predict_tokens(batch))
    return [pred for pred in inferences]
