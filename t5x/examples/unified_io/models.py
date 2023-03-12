# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5X Models.

This module uses layers.py to build a higher-level model structure and define
methods for the loss computation as well as a train, prediction, and evaluation
steps.
"""

import abc
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Type, Union

import clu.metrics as clu_metrics
from flax import core as flax_core
from flax import linen as nn
from t5x import optimizers as optim
from flax.core import scope as flax_scope
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import seqio

from t5x.examples.unified_io import decoding
from t5x import losses
from t5x import metrics as metrics_lib
import tensorflow as tf
import typing_extensions

from .data.data_utils import UnifiedIOFeatureConverter
from t5x.examples.unified_io import decoding
from ...losses import cross_entropy_with_logits

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
Optimizer = optim.Optimizer
PyTreeDef = type(jax.tree_structure(None))


class TokensIdsToLogitsCallable(typing_extensions.Protocol):
  """Token ids to logits mapping call signature."""

  def __call__(
      self, token_ids: jnp.ndarray, cache: Mapping[str, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Performs forward pass to convert token ids to logits.

    Args:
      token_ids: [batch_size, 1] int32 tokens for single position used during
        incremental decoding. Non-0 prefix tokens to be used as a forced prompt.
      cache: flax attention cache.

    Returns:
      a tuple of logits with a shape [batch_size, vocab_size] and an updated
      cache.
    """
    ...

class DecodeFnCallable(typing_extensions.Protocol):
  """Decoding function call signature."""

  def __call__(self, *, inputs: jnp.ndarray, cache: Mapping[str, jnp.ndarray],
               tokens_to_logits: TokensIdsToLogitsCallable, eos_id: int,
               num_decodes: int, decode_rng: Optional[jnp.ndarray],
               **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Decoding function interface.

    Args:
      inputs: [batch_size, max_decode_len] int32 sequence of tokens, with non-0
        prefix tokens to be used as a forced prompt.
      cache: flax attention cache.
      tokens_to_logits: fast autoregressive decoder function taking single token
        slices and cache and returning next-token logits and updated cache.
      eos_id: end-of-sentence token for target vocabulary.
      num_decodes: number of decoded sequences to be returned.
      decode_rng: an optional JAX PRNG Key for stochastic sampling routines.
      **kwargs: an optional kwargs. One common usecase of this is passing
        decoding parameters at the callsite.

    Returns:
      decodes: Array of sequences: [batch_size, num_decodes, max_decode_len].
        The `num_decodes` dimension is expected to be sorted by the `scores`,
        i.e., `decodes[:, -1, :] has the highest scores among `num_decodes`
        decoded sequences.
      scores: Array of log likelihood scores: [batch_size, num_decodes]
    """
    ...


class BaseModel(abc.ABC):
  """Abstract base class for models.

  Subclasses must implement the abstract methods. Any additional arguments added
  to these methods must have defaults or be bound at run time to fit the
  interface expected by the standard training, inference, and evaluation
  functions.
  """

  FEATURE_CONVERTER_CLS: Type[seqio.FeatureConverter]

  def __init__(self, optimizer_def: optim.OptimizerDef):
    # TODO(jbulian): Move the optimizer out of the model and make it a training
    #                parameter.
    self.optimizer_def = optimizer_def

  @abc.abstractmethod
  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, MetricsMap]]:
    """Computes loss and metrics.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    pass

  def eval_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, MetricsMap]]:
    """Computes loss and metrics during the evaluation.

    Args:
      params: model parameters.
      batch: a batch of inputs.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    return self.loss_fn(
        params=params,
        batch=batch,
        dropout_rng=None,
    )

  def predict_batch(self, params: PyTreeDef,
                    batch: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Thin wrapper around `self.predict_batch_with_aux`."""
    # The first element of the return value is the predicted sequences.
    return self.predict_batch_with_aux(params, batch)[0]

  @abc.abstractmethod
  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predicts batch with auxiliary outputs."""
    pass

  @abc.abstractmethod
  def score_batch(self,
                  params: PyTreeDef,
                  batch: Mapping[str, jnp.ndarray],
                  return_intermediates: bool = False) -> jnp.ndarray:
    """Computes scores for batch."""
    pass

  @abc.abstractmethod
  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Returns the initial variables of the model."""
    pass

  def get_initial_metrics(self) -> MetricsMap:
    """Dictionary of metrics and initial values."""
    return {}

  # TODO(cpgaffney) clean up summarize_metrics_fn
  def summarize_metrics_fn(self, metrics: MetricsMap, duration: float,
                           num_steps: int) -> Mapping[str, Array]:
    """Converts metrics into tensorboard-friendly summary.

    Args:
      metrics: Metrics obtained from `loss_fn`, summed across multiple batches.
      duration: The duration of the run being summarized.
      num_steps: The number of steps the metrics are summed across.

    Returns:
      summary: Metrics in tensorboard friendly format.
    """
    del duration, num_steps
    return {k: v.compute() for k, v in metrics.items()}


# Sentinel used instead of None to indicate missing values. For backward
# compatibility purposes; will be removed in an upcoming revision.
_NoValueSentinel = object()


class BaseTransformerModel(BaseModel):
  """Abstract base class for Transformer models using.

  Subclasses must implement `predict_batch_with_aux`, `score_batch`,
  `get_initial_variables` from `BaseModel` as well as `_compute_logits`.
  """

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      decode_fn: Optional[DecodeFnCallable] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      loss_normalizing_by_weight_sum: Optional[bool] = False,
      text_decoder_length = None,
      image_decoder_length = None,
  ):
    self.module = module
    self._input_vocabulary = input_vocabulary
    self._output_vocabulary = output_vocabulary
    self._decode_fn = decode_fn
    self._label_smoothing = label_smoothing
    self._z_loss = z_loss
    self._loss_normalizing_factor = loss_normalizing_factor
    self._loss_normalizing_by_weight_sum = loss_normalizing_by_weight_sum
    self._text_decoder_length = text_decoder_length
    self._image_decoder_length = image_decoder_length

    super().__init__(optimizer_def=optimizer_def)

  @property
  def input_vocabulary(self):
    return self._input_vocabulary

  @property
  def output_vocabulary(self):
    return self._output_vocabulary

  @property
  def decode_fn(self):
    return self._decode_fn

  @abc.abstractmethod
  def _compute_logits(self,
                      params: PyTreeDef,
                      batch: Mapping[str, jnp.ndarray],
                      dropout_rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Computes logits via a forward pass of the model."""
    pass

  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: Optional[float] = None,
      z_loss: Optional[float] = None,
      loss_normalizing_factor: Union[Optional[float], object] = _NoValueSentinel,
      loss_normalizing_by_weight_sum:  Union[Optional[float], object] = _NoValueSentinel,
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, MetricsMap]]:
    """Loss function used for training with a cross-entropy loss."""

    # Default these to the constructor values. In the future, they may be
    # removed as parameters for `loss_fn`.
    label_smoothing = (
        self._label_smoothing if label_smoothing is None else label_smoothing)
    z_loss = self._z_loss if z_loss is None else z_loss
    if loss_normalizing_factor is _NoValueSentinel:
      loss_normalizing_factor = self._loss_normalizing_factor

    if loss_normalizing_by_weight_sum is _NoValueSentinel:
      loss_normalizing_by_weight_sum = self._loss_normalizing_by_weight_sum

    text_logits, image_logits, image_decoder_targets = self._compute_logits(params, batch, dropout_rng)

    text_weights = batch.get('text_decoder_masks', None)
    image_weights = batch.get('image_target_loss_masks', None)

    text_loss, text_z_loss, text_weight_sum = losses.compute_weighted_cross_entropy(
        text_logits,
        targets=batch['text_decoder_targets'],
        weights=text_weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
        loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
        vocab_size = 33152,
        modality = 'text')

    image_loss, image_z_loss, image_weight_sum = losses.compute_weighted_cross_entropy(
        image_logits,
        targets=image_decoder_targets,
        weights=image_weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
        loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
        vocab_size = 16384,
        modality = 'image')

    metrics = compute_base_metrics(
        logits=text_logits,
        image_logits=image_logits,
        targets=batch['text_decoder_targets'],
        image_targets=image_decoder_targets,
        mask=text_weights,
        image_mask=image_weights,
        loss=text_loss,
        image_loss=image_loss,
        z_loss=text_z_loss,
        image_z_loss=image_z_loss)

    loss = text_loss + image_loss
    weight_sum = text_weight_sum + image_weight_sum
    return loss, metrics

  def predict_batch_with_cls(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      num_option: int = 4,
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, MetricsMap]]:
    """Loss function used for training with a cross-entropy loss."""

    text_encoder_inputs = batch['text_encoder_inputs']
    image_encoder_inputs = batch['image_encoder_inputs']
    text_encoder_masks = batch['text_encoder_masks']
    image_input_masks = batch['image_input_masks']

    text_decoder_inputs = batch['text_decoder_inputs']
    text_decoder_targets = batch['text_decoder_targets']
    text_decoder_masks = batch['text_decoder_masks']
    text_decoder_segment_ids = batch['text_decoder_segment_ids']
    text_decoder_positions = batch['text_decoder_positions']
    image_target_masks = batch['image_target_masks']

    batch_size = text_decoder_inputs.shape[0]
    # Default these to the constructor values. In the future, they may be
    # removed as parameters for `loss_fn`.
    encoded_inputs, encoder_masks = self.module.apply({'params': params},
        text_encoder_inputs,
        image_encoder_inputs,
        text_encoder_masks,
        image_input_masks,
        image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
        text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
        enable_dropout=False,
        method=self.module.encode)

    if num_option is None:
      num_option = tf.shape(text_decoder_inputs)[1]

    encoded, encoder_position_embedding = encoded_inputs
    encoded = decoding.flat_batch_beam_expand(encoded, num_option)
    encoder_position_embedding = decoding.flat_batch_beam_expand(encoder_position_embedding, num_option)
    encoder_masks = decoding.flat_batch_beam_expand(encoder_masks, num_option)
    encoded_inputs = (encoded, encoder_position_embedding)

    image_decoder_inputs = jnp.zeros([encoded.shape[0], 1], jnp.int32)
    image_decoder_targets = jnp.zeros([encoded.shape[0], 1], jnp.int32)
    image_decoder_masks = jnp.zeros([encoded.shape[0], 1], jnp.int32)

    text_decoder_inputs = jnp.reshape(text_decoder_inputs, [batch_size*num_option, -1])
    text_decoder_targets = jnp.reshape(text_decoder_targets, [batch_size*num_option, -1])
    text_decoder_masks = jnp.reshape(text_decoder_masks, [batch_size*num_option, -1])
    text_decoder_segment_ids = jnp.reshape(text_decoder_segment_ids, [batch_size*num_option, -1])
    text_decoder_positions = jnp.reshape(text_decoder_positions, [batch_size*num_option, -1])

    text_logits, image_logits, image_decoder_targets = self.module.apply({'params': params},
        encoded_inputs,
        encoder_masks,
        text_decoder_inputs,
        image_decoder_inputs,
        text_decoder_targets,
        image_decoder_targets,
        text_decoder_masks=text_decoder_masks,
        image_decoder_masks=image_decoder_masks,
        text_decoder_segment_ids=text_decoder_segment_ids,
        text_decoder_positions=text_decoder_positions,
        enable_dropout=False,
        method=self.module.decode)

    text_weights =text_decoder_masks
    image_weights = image_decoder_masks
    z_loss = self._z_loss

    text_loss, text_z_loss, text_weight_sum = losses.compute_weighted_cross_entropy(
        text_logits,
        targets=text_decoder_targets,
        weights=text_weights,
        label_smoothing=False,
        z_loss=z_loss,
        loss_normalizing_factor=1.0,
        loss_normalizing_by_weight_sum=True,
        return_sum=False,
        vocab_size = 33152,
        modality = 'text')

    image_loss, image_z_loss, image_weight_sum = losses.compute_weighted_cross_entropy(
        image_logits,
        targets=image_decoder_targets,
        weights=image_weights,
        label_smoothing=False,
        z_loss=z_loss,
        loss_normalizing_factor=1.0,
        loss_normalizing_by_weight_sum=True,
        return_sum=False,
        vocab_size = 16384,
        modality = 'image')

    loss = jnp.sum(text_loss, axis=1) + jnp.sum(image_loss, axis=1)
    loss = jnp.reshape(loss, [batch_size, -1])
    # predicted = jnp.argmin(loss, axis=1)
    fake_output = jnp.zeros([batch_size, 1], jnp.int32)

    return fake_output, dict(scores=loss)

  # TODO(cpgaffney) Modify when all users are able to use compute_base_metrics
  def _compute_metrics(
      self,
      batch: Mapping[str, jnp.ndarray],
      logits: jnp.ndarray,
      loss: jnp.ndarray,
      total_z_loss: jnp.ndarray,
      weight_sum: jnp.ndarray,
  ) -> MetricsMap:
    """Compute metrics given the logits, targets and loss."""
    additional_metrics = {
        'z_loss': metrics_lib.Sum.from_model_output(total_z_loss),
        'cross_ent_loss': metrics_lib.Sum.from_model_output(loss - total_z_loss)
    }
    return compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        weights=batch.get('decoder_loss_weights', None),
        loss=loss,
        weight_sum=weight_sum,
        additional_metrics=additional_metrics)

  def get_initial_metrics(self):
    return {}


class EncoderDecoderModel(BaseTransformerModel):
  """Wrapper class for the models.Transformer nn.module."""

  FEATURE_CONVERTER_CLS = UnifiedIOFeatureConverter
  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      decode_fn: DecodeFnCallable = decoding.beam_search,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      loss_normalizing_by_weight_sum: Optional[bool] = False,
      text_decoder_length = None,
      image_decoder_length = None,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
        loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
        text_decoder_length = text_decoder_length,
        image_decoder_length = image_decoder_length,
    )

  # Adds explicit loss method for proper configuration.
  # TODO(b/194404217): Remove once gin correctly handles child class configs.
  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: Optional[float] = None,
      z_loss: Optional[float] = None,
      loss_normalizing_factor: Union[Optional[float],
                                     object] = _NoValueSentinel,
      loss_normalizing_by_weight_sum: Union[Optional[float],
                                     object] = _NoValueSentinel,
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, MetricsMap]]:

    return super().loss_fn(
        params=params,
        batch=batch,
        dropout_rng=dropout_rng,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
        loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    text_encoder_shape = input_shapes['text_encoder_inputs']
    text_encoder_type = input_types.get('text_encoder_inputs', jnp.float32)
    image_encoder_shape = input_shapes['image_encoder_inputs']
    image_encoder_type = input_types.get('image_encoder_inputs', jnp.float32)
    text_decoder_shape = input_shapes['text_decoder_inputs']
    text_decoder_type = input_types.get('text_decoder_inputs', jnp.float32)
    image_decoder_shape = input_shapes['image_decoder_targets']
    image_decoder_type = input_types.get('image_decoder_targets', jnp.float32)
    initial_variables = self.module.init(
        rng,
        jnp.ones(text_encoder_shape, text_encoder_type),
        jnp.ones(image_encoder_shape, image_encoder_type),
        jnp.ones(text_decoder_shape, text_decoder_type),
        jnp.ones(image_decoder_shape, image_decoder_type),
        jnp.ones(text_decoder_shape, text_decoder_type),
        decode=False,
        enable_dropout=False,
        cache_text_length=self._text_decoder_length,
        cache_image_length=self._image_decoder_length,
        vae_decode=True)
    return initial_variables

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
      mutable: flax_scope.CollectionFilter = False
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None

    return self.module.apply(
        {'params': params},
        batch['text_encoder_inputs'],
        batch['image_encoder_inputs'],
        batch['text_decoder_inputs'],
        batch['image_decoder_targets'],
        batch['text_decoder_targets'],
        text_encoder_masks=batch.get('text_encoder_masks', None),
        image_encoder_masks=batch.get('image_input_masks', None),
        image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
        text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
        text_decoder_masks=batch.get('text_decoder_masks', None),
        image_decoder_masks=batch.get('image_target_masks', None),
        text_decoder_segment_ids=batch.get('text_decoder_segment_ids', None),
        text_decoder_positions=batch.get('text_decoder_positions', None),
        cache_text_length=self._text_decoder_length,
        cache_image_length=self._image_decoder_length,
        decode=False,
        enable_dropout=rngs is not None,
        rngs=rngs,
        mutable=mutable)

  def predict_with_answer_options(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
  ):
    _encoded_inputs, _encoder_masks = self.module.apply(
      {'params': params},
      batch['text_encoder_inputs'],
      batch['image_encoder_inputs'],
      batch['text_encoder_masks'],
      batch['image_input_masks'],
      image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
      text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
      enable_dropout=False,
      method=self.module.encode
    )

    all_losses = []
    n_options = batch["output_options"].shape[1]
    max_options = 800
    n_groups = (n_options + max_options - 1) // max_options
    for i in range(n_groups):
      output_options = batch["output_options"][:, i*max_options:(i+1)*max_options]
      batch_size, num_option = output_options.shape[:2]
      encoded, encoder_position_embedding = _encoded_inputs
      encoded = decoding.flat_batch_beam_expand(encoded, num_option)
      encoder_position_embedding = decoding.flat_batch_beam_expand(encoder_position_embedding, num_option)
      encoder_masks = decoding.flat_batch_beam_expand(_encoder_masks, num_option)
      encoded_inputs = (encoded, encoder_position_embedding)

      decoded_size = batch_size*num_option

      # `output_options` does not have EOS or BOS, we need to do a bit work to correctly-formatted
      # text inputs/outputs here
      text_decoder_inputs = output_options.reshape((decoded_size, -1))
      text_decoder_targets = text_decoder_inputs
      text_decoder_targets = jnp.pad(text_decoder_targets, [[0, 0], [0, 1]])  # Add room for EOS

      text_decoder_masks = text_decoder_inputs > 0
      text_decoder_inputs = jnp.pad(text_decoder_inputs, [[0, 0], [1, 0]])
      text_decoder_masks = jnp.pad(text_decoder_masks, [[0, 0], [1, 0]], constant_values=True)

      eos_mask = jnp.logical_and(text_decoder_masks, text_decoder_targets == 0)
      text_decoder_targets = text_decoder_targets + eos_mask

      # from jax.experimental import host_callback as hcb
      # hcb.id_print(text_decoder_inputs[:5], targets=text_decoder_inputs[:5])
      # hcb.id_print(text_decoder_targets[:5], inputs=text_decoder_targets[:5])
      # hcb.id_print(text_decoder_masks[:5], mask=text_decoder_masks[:5])

      # Dummy image values
      image_decoder_inputs = jnp.zeros([encoded.shape[0], 1], jnp.int32)
      image_decoder_targets = jnp.zeros([encoded.shape[0], 1], jnp.int32)
      image_decoder_masks = jnp.zeros([encoded.shape[0], 1], jnp.int32)

      text_logits, image_logits, image_decoder_targets = self.module.apply(
        {'params': params},
        encoded_inputs,
        encoder_masks,
        text_decoder_inputs,
        image_decoder_inputs,
        text_decoder_targets,
        image_decoder_targets,
        text_decoder_masks=text_decoder_masks,
        image_decoder_masks=image_decoder_masks,
        enable_dropout=False,
        method=self.module.decode
      )

      text_loss, text_z_loss, text_weight_sum = losses.compute_weighted_cross_entropy(
        text_logits,
        targets=text_decoder_targets,
        weights=text_decoder_masks,
        label_smoothing=False,
        loss_normalizing_factor=1.0,
        loss_normalizing_by_weight_sum=True,
        return_sum=False,
        vocab_size = 33152,
        modality = 'text'
      )

      text_loss = jnp.sum(text_loss, axis=1)
      text_loss = jnp.reshape(text_loss, [batch_size, -1])
      all_losses.append(text_loss)

    text_loss = jnp.concatenate(all_losses, -1)
    selected_option_ix = jnp.argmin(text_loss, -1)
    ix = jnp.arange(0, len(selected_option_ix))
    selected_options = output_options[ix, selected_option_ix]
    text_loss = text_loss[ix, selected_option_ix]

    return selected_options, {'scores': text_loss}

  def _compute_logits_from_slice(
      self, flat_ids: jnp.ndarray, flat_cache: Mapping[str, jnp.ndarray], cur_index: int,
      live_seqs: jnp.ndarray, params: PyTreeDef, encoded_inputs: jnp.ndarray, encoder_masks: jnp.ndarray,
      text_length: int, image_length: int, logit_masks: jnp.ndarray = None) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]

    def update_flat_ids(x):
      x = jnp.zeros_like(x) + self.module.config.vocab_size - 1
      return x

    def update_pos_ids(x):
      x = x + self.module.config.max_text_length - text_length
      return x

    def identity_fn(x):
      return x

    def update_ones(x):
      x = jnp.zeros_like(x) + 1
      return x

    def update_zeros(x):
      x = jnp.zeros_like(x)
      return x

    flat_ids = jax.lax.cond(
        jax.lax.eq(cur_index, text_length),
        lambda: update_flat_ids(flat_ids),
        lambda: identity_fn(flat_ids))

    seg_ids = jax.lax.cond(
        jax.lax.ge(cur_index, text_length),
        lambda: update_ones(flat_ids),
        lambda: update_zeros(flat_ids))

    decoder_masks = jax.lax.cond(cur_index < text_length,
        lambda: jnp.reshape((live_seqs == 1).sum(axis=-1) == 0, (-1,1)),
        lambda: jnp.ones(flat_ids.shape, dtype=jnp.bool_))

    flat_logits, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        encoder_masks,  # only needed for encoder padding mask
        flat_ids,
        decoder_masks=decoder_masks,
        decoder_segments=seg_ids,
        enable_dropout=False,
        decode=True,
        image_decode_length=image_length,
        text_decode_length=text_length,
        cur_index=cur_index,
        mutable=['cache'],
        method=self.module.sample)
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']

    cfg = self.module.config
    total_vocab_size = cfg.vocab_size + cfg.image_vocab_size
    logit_range = jnp.reshape(jnp.arange(total_vocab_size), [1, 1, -1])
    image_logits_mask = jnp.reshape(logit_range < cfg.vocab_size, [1, -1])
    text_logits_mask = jnp.reshape(logit_range >= cfg.vocab_size, [1, -1])

    flat_logits = jax.lax.cond(
        jax.lax.ge(cur_index, text_length),
        lambda: jnp.where(image_logits_mask, -1e10, flat_logits),
        lambda: jnp.where(text_logits_mask, -1e10, flat_logits))

    def update_mask(flat_logits, logit_masks, cur_index):
      mask = jnp.reshape(logit_masks[cur_index], [1, -1])
      flat_logits = jnp.where(mask, -1e10, flat_logits)
      return flat_logits

    # apply mask here.
    if logit_masks is not None:
      flat_logits = jax.lax.cond(
        jax.lax.lt(cur_index, logit_masks.shape[0]),
        lambda: update_mask(flat_logits, logit_masks, cur_index),
        lambda: identity_fn(flat_logits))

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      text_length = 64,
      image_length = 256,
      logit_mask_fn = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with fast decoding beam search on a batch.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    This method can be used with a customizable decoding function as long as it
    follows the signature of `DecodeFnCallable`. In order to provide a unified
    interface for the decoding functions, we use a generic names. For example a
    beam size is a concept unique to beam search. Conceptually, it corresponds
    to the number of sequences returned by the beam search.  Therefore, the
    generic argument `num_decodes` corresponds to the beam size if
    `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
    corresponds to the number of indepedent sequences to be sampled. Typically
    `num_decodes = 1` is used for tempeature sampling.

    If `return_all_decodes = True`, the return tuple contains the predictions
    with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
    probability of the generated sequence) with a shape [batch, num_decodes].

    If `return_all_decodes = False`, the return tuple contains the predictions
    with a shape [batch, max_decode_len] and the scores with a shape [batch].

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    if "output_options" in batch:
      return self.predict_with_answer_options(params, batch)

    # [batch, input_len]
    text_encoder_inputs = batch['text_encoder_inputs']
    image_encoder_inputs = batch['image_encoder_inputs']
    text_encoder_masks = batch['text_encoder_masks']
    image_input_masks = batch['image_input_masks']

    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    text_encoder_shape = batch['text_encoder_inputs'].shape
    text_encoder_type = batch['text_encoder_inputs'].dtype
    image_encoder_shape = batch['image_encoder_inputs'].shape
    image_encoder_type = batch['image_encoder_inputs'].dtype
    text_decoder_shape = batch['text_decoder_inputs'].shape
    text_decoder_type = batch['text_decoder_inputs'].dtype
    image_decoder_shape = batch['image_decoder_targets'].shape
    image_decoder_type = batch['image_decoder_targets'].dtype

    _, variables_with_cache = self.module.apply(
        {'params': params},
        jnp.ones(text_encoder_shape, text_encoder_type),
        jnp.ones(image_encoder_shape, image_encoder_type),
        jnp.ones(text_decoder_shape, text_decoder_type),
        jnp.ones(image_decoder_shape, image_decoder_type),
        jnp.ones(text_decoder_shape, text_decoder_type),
        image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
        text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
        decode=True,
        enable_dropout=False,
        vae_decode=False,
        cache_text_length=text_length,
        cache_image_length=image_length,
        mutable=['cache'])

    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs, encoder_masks = self.module.apply({'params': params},
        text_encoder_inputs,
        image_encoder_inputs,
        text_encoder_masks,
        image_input_masks,
        image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
        text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
        enable_dropout=False,
        method=self.module.encode)

    encoded, encoder_position_embedding = encoded_inputs
    encoded = decoding.flat_batch_beam_expand(encoded, num_decodes)
    encoder_masks = decoding.flat_batch_beam_expand(encoder_masks, num_decodes)
    encoded_inputs = (encoded, encoder_position_embedding)

    if logit_mask_fn is not None:
      logit_masks = logit_mask_fn()
    else:
      logit_masks = None

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        encoded_inputs=encoded_inputs,
        encoder_masks=encoder_masks,
        text_length=text_length,
        image_length=image_length,
        logit_masks=logit_masks)

    if decoder_params is None:
      decoder_params = {}

    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decod_prompt_inputs` will be filled with the sampled ids.
    batch_size = image_decoder_shape[0]
    decoder_prompt_inputs = jnp.zeros([batch_size, text_length+image_length], text_decoder_type)

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if self._decode_fn == decoding.beam_search:
      decodes, scores, logprobs = self._decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          alpha=0.0,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=self.output_vocabulary.eos_id,
          num_decodes=num_decodes,
          cache_offset=1 if scanned else 0,
          input_texts=text_encoder_inputs,
          **decoder_params)
    else:
      # TODO support logprobs in non-beam search decoding
      decodes, scores, logprobs = self._decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=self.output_vocabulary.eos_id,
          num_decodes=num_decodes,
          topk = 0,
          topp = 0.9,
          cache_offset=1 if scanned else 0,
          input_texts=text_encoder_inputs,
          **decoder_params)

    scores = jax.lax.stop_gradient(scores)
    if logprobs is not None:
      logprobs = jax.lax.stop_gradient(logprobs)

    if image_length == 256:
      if return_all_decodes:
        image_decodes = decodes[:, :, -256:].reshape(-1, 256)
      else:
        image_decodes = decodes[:, -1, -256:]
      image_decodes = image_decodes - self.module.config.vocab_size
      decodes = decodes[:, :, :-256]
      img = self.module.apply(
          {'params': params},
          method=self.module.decode_code,
          code_b=image_decodes)

      if return_all_decodes:
        img = jnp.reshape(img, decodes.shape[:2] + img.shape[1:])
        image_decodes = jnp.reshape(image_decodes, decodes.shape[:2] + image_decodes.shape[1:])

      if return_all_decodes:
        return decodes, dict(img=img, scores=scores, logprobs=logprobs, img_tokens=image_decodes)
      else:
        return decodes[:, -1, :], dict(img=img, scores=scores[:, -1], logprobs=None if logprobs is None else logprobs[:,-1], img_tokens=image_decodes[:, -1])

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores, 'logprobs':logprobs}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1], 'logprobs': None if logprobs is None else logprobs[:,-1]}

  def score_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      return_sum: bool = True
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""
    text_logits, image_logits, image_decoder_targets = self._compute_logits(params, batch, None)
    text_decoder_targets = batch['text_decoder_targets']

    text_weights = batch.get('text_decoder_masks', None)
    if text_weights is None:
      text_weights = jnp.where(text_decoder_targets == 0,
                               jnp.zeros_like(text_decoder_targets),
                               jnp.ones_like(text_decoder_targets))
    image_weights = batch.get('image_target_masks', None)

    token_scores = cross_entropy_with_logits(
      text_logits,
      common_utils.onehot(text_decoder_targets, text_logits.shape[-1], on_value=1, off_value=0),
      z_loss=0.0
    )[0] * text_weights

    sequence_scores = -token_scores.sum(-1) / text_weights.sum(-1)

    return sequence_scores


@jax.vmap
def remove_prefix(sequence: jnp.ndarray,
                  prefix_length: jnp.ndarray) -> jnp.ndarray:
  """Remove the prefix portion and shift to the left by the prefix length.

  The example below uses non-decorated function definition, i.e., arrays do not
  have batch dimension. `jax.vmap` internally inserts the batch dimension at
  axis=0. The shape annotations do not include the batch dimension either.

  Example:
  ```python
  sequence = [1, 2, 3, 4, 5, 6, 7, 0]
  prefix_length = 2
  remove_prefix(sequence, prefix_length) = [3, 4, 5, 6, 7, 0, 0, 0]
  ```

  Note that this function assumes that the padding token has an id of 0.

  Args:
    sequence: [length] array.
    prefix_length: scalar, i.e., rank 0 array.

  Returns:
    [length] array with the prefix removed and the suffix shifted.
  """
  length = sequence.shape[-1]
  # A binary mask with 1 at inputs.
  inputs_mask = (jnp.arange(length) < prefix_length)
  # A binary mask with 1 at the targets and padding positions.
  targets_and_padding_mask = jnp.logical_not(inputs_mask).astype(sequence.dtype)
  # Since padding id = 0, the padding mask is zeroed out.
  targets = sequence * targets_and_padding_mask
  # Shift to the left by prefix length. Wrapped elements are already zeroed.
  return jnp.roll(targets, -prefix_length, axis=-1)


# TODO(cpgaffney) Remove this method when dependencies no longer use - rely on
# WeightedAccuracy Metric instead.
def compute_weighted_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
   weights: None or array of shape [batch, length]

  Returns:
    Scalar accuracy.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  if weights is not None:
    accuracy = accuracy * weights

  return jnp.sum(accuracy)


# TODO(cpgaffney) remove when users rely on compute_base_metrics
def compute_metrics(logits: jnp.ndarray, targets: jnp.ndarray,
                    weights: jnp.ndarray, loss: jnp.ndarray,
                    weight_sum: jnp.ndarray,
                    additional_metrics: MetricsMap) -> MetricsMap:
  """Compute summary metrics."""
  accuracy = compute_weighted_accuracy(logits, targets, weights)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'weight_sum': weight_sum,
      'num_examples': targets.shape[0],
      'num_tokens': targets.size
  }
  metrics = metrics_lib.create_metrics_dict(metrics)
  metrics.update(additional_metrics)
  return metrics


def compute_base_metrics(
    logits: jnp.ndarray,
    image_logits: jnp.ndarray,
    targets: jnp.ndarray,
    image_targets: jnp.ndarray,
    mask: jnp.ndarray,
    image_mask: jnp.ndarray,
    loss: jnp.ndarray,
    image_loss: jnp.ndarray,
    z_loss: Optional[jnp.ndarray] = None,
    image_z_loss: Optional[jnp.ndarray] = None,
) -> MetricsMap:
  """Compute summary metrics.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
   mask: None or array of shape [batch, length]. Note: must consist of boolean
     values (float-valued weights not supported).
   loss: loss (float)
   z_loss: z_loss (float)

  Returns:
    Dict of metrics.
  """
  num_examples = targets.shape[0]
  num_tokens = targets.size
  num_devices = jax.device_count()
  assert num_devices, 'JAX is reporting no devices, but it should.'
  # Note: apply mask again even though mask has already been applied to loss.
  # This is needed to divide by mask sum, but should not affect correctness of
  # the numerator.
  nonpadding_tokens = np.prod(targets.size)
  if mask is not None:
    nonpadding_tokens = jnp.sum(mask)
  metrics = {
      'accuracy':
          clu_metrics.Accuracy.from_model_output(
              logits=logits, labels=targets.astype(jnp.int32), mask=mask),
      'image_accuracy':
          clu_metrics.Accuracy.from_model_output(
              logits=image_logits, labels=image_targets.astype(jnp.int32), mask=image_mask),
      'loss':
          metrics_lib.AveragePerStep(total=loss),
      'image_loss':
          metrics_lib.AveragePerStep(total=image_loss),
      'timing/seqs_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples),
      'timing/steps_per_second':
          metrics_lib.StepsPerTime.from_model_output(),
      'timing/seconds':
          metrics_lib.Time(),
      'timing/seqs':
          metrics_lib.Sum(num_examples),
      'timing/seqs_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples /
                                                 num_devices),
      'timing/target_tokens_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_tokens),
      'timing/target_tokens_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_tokens /
                                                 num_devices),
      'nonpadding_fraction':
          clu_metrics.Average(total=nonpadding_tokens, count=num_tokens),
  }
  if z_loss is not None:
    metrics.update({
        'z_loss':
            metrics_lib.AveragePerStep(total=z_loss),
        'image_z_loss':
            metrics_lib.AveragePerStep(total=image_z_loss),
        'z_loss_per_all_target_tokens':
            clu_metrics.Average(total=z_loss, count=num_tokens),
        'cross_ent_loss':
            metrics_lib.AveragePerStep(total=loss - z_loss),
        'image_cross_ent_loss':
            metrics_lib.AveragePerStep(total=image_loss - image_z_loss),
        'cross_ent_loss_per_all_target_tokens':
            clu_metrics.Average(total=jnp.sum(loss - z_loss), count=num_tokens)
    })
  return metrics


def get_input_vocabulary(model: BaseTransformerModel) -> seqio.Vocabulary:
  return model.input_vocabulary

def get_output_vocabulary(model: BaseTransformerModel) -> seqio.Vocabulary:
  return model.output_vocabulary
