import functools
import os.path

import gin
import jax
import tensorflow as tf

import t5x
from t5x import decoding
from t5x.examples.t5.network import T5Config, Transformer
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from t5x.models import EncoderDecoderModel


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
                                         )
                                         # ngram_block_size=ngram_block_size,
                                         # context_ngram_block_size=context_ngram_block_size,
                                         # min_beam_length=min_beam_length)
    elif decode_fn == "temperature":
      self.decode_fn = functools.partial(decoding.temperature_sample,
                                         temperature=0.3,
                                         topk=5,
                                         topp=0.0,
                                         )
                                         # ngram_block_size=ngram_block_size,
                                         # context_ngram_block_size=context_ngram_block_size,
                                         # min_beam_length=min_beam_length)
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
      return self.model.predict_batch_with_aux(
        params, batch,
        num_decodes=5,
        return_all_decodes=False,
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
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if model_type == "base":
      gin_files = [os.path.join(abs_path, './t5_1_1/base.gin')]
    elif model_type == "large":
      gin_files = [os.path.join(abs_path, './t5_1_1/large.gin')]
    elif model_type == "xl":
      gin_files = [os.path.join(abs_path, './t5_1_1/xl.gin')]
    else:
      raise NotImplementedError

    self.task_feature_lengths = {'inputs': self.text_inputs,
                                 'targets': self.text_targets,
                                 }
    self._parse_gin(gin_files)

  @property
  def input_shapes(self):
    return {
      "encoder_input_tokens": (self.batch_size, self.text_inputs),
      "decoder_input_tokens": (self.batch_size, self.text_targets),
      "decoder_target_tokens": (self.batch_size, self.text_targets),
      "decoder_loss_weights": (self.batch_size, self.text_targets),
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
      'DROPOUT_RATE = 0.0',
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
        gin_files, gin_bindings, finalize_config=False)

  def _load_model(self, model_type):
    """Load up a T5X `Model` after parsing training gin config."""
    config = gin.get_configurable(T5Config)()
    _module = Transformer(config=config)
    return EncoderDecoderModel(
      module=_module,
      input_vocabulary=self.vocabulary,
      output_vocabulary=self.vocabulary,
      decode_fn=self.decode_fn,
      optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
    )

  def build_dataset(self, dialogue_input):
    prompt = dialogue_input.split("\t")[0]
    dialogue_input = dialogue_input.split("\t")[1]
    dialogue_turns = " <turn> ".join([turn.strip() for turn in dialogue_input.split("<sep>")])
    context = prompt + " <sep> " + dialogue_turns
    text_inputs = self.vocabulary.encode_tf(context)
    ds = tf.data.Dataset.from_tensors({
      'inputs': text_inputs,
      'targets': self.vocabulary.encode_tf(' '),
    }).repeat(self.batch_size)
    return ds

  def __call__(self, dialogue_input, debug: bool = False):
    ds = self.build_dataset(dialogue_input)
    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
      ds, task_feature_lengths=self.task_feature_lengths)
    model_ds = model_ds.batch(self.batch_size)

    inferences = (results for batch in model_ds.as_numpy_iterator()
                  for results in self.predict_tokens(batch))
    return [pred for pred in inferences]
