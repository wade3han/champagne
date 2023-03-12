import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import seqio
from flax import linen as nn
from flax.core import scope as flax_scope

from t5x import optimizers as optim, losses
from t5x.examples.unified_io import decoding
from .data.data_utils import UnifiedIOFeatureConverter, UnifiedIOPairFeatureConverter
from .models import DecodeFnCallable, Array, PyTreeDef, EncoderDecoderModel, _NoValueSentinel, compute_base_metrics
from ...metrics import MetricsMap


class MultiImageEncoderDecoderModel(EncoderDecoderModel):
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
      text_decoder_length=None,
      image_decoder_length=None,
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
      text_decoder_length=text_decoder_length,
      image_decoder_length=image_decoder_length,
    )

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

  def _compute_logits_from_slice(
      self, flat_ids: jnp.ndarray, flat_cache: Mapping[str, jnp.ndarray], cur_index: int,
      live_seqs: jnp.ndarray, params: PyTreeDef, encoded_inputs: jnp.ndarray, encoder_masks: jnp.ndarray,
      text_length: int, image_length: int, logit_masks: jnp.ndarray = None) -> Tuple[
    jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""

    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]

    def update_flat_ids(x):
      x = jnp.zeros_like(x) + self.module.config.vocab_size - 1
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
                                 lambda: jnp.reshape((live_seqs == 1).sum(axis=-1) == 0, (-1, 1)),
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
      text_length=64,
      image_length=256,
      logit_mask_fn=None,
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
    decoder_prompt_inputs = jnp.zeros([batch_size, text_length + image_length], text_decoder_type)

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    try:
      # TODO support logprobs in non-beam search decoding
      decodes, scores, logprobs = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        topk=0,
        topp=0.9,
        cache_offset=1 if scanned else 0,
        input_texts=text_encoder_inputs,
        **decoder_params)
    except TypeError:
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
        return decodes[:, -1, :], dict(img=img, scores=scores[:, -1],
                                       logprobs=None if logprobs is None else logprobs[:, -1],
                                       img_tokens=image_decodes[:, -1])

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores, 'logprobs': logprobs}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1], 'logprobs': None if logprobs is None else logprobs[:, -1]}

  def predict_interactive(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      decode_fn,
      decode_fn_name: str,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      text_length=64,
      image_length=256,
      logit_mask_fn=None,
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
    decoder_prompt_inputs = jnp.zeros([batch_size, text_length + image_length], text_decoder_type)

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if decode_fn_name == "beam":
      decodes, scores, logprobs = decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        alpha=0.0,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        input_texts=text_encoder_inputs,
        **decoder_params)
    elif decode_fn_name == "temperature":
      # TODO support logprobs in non-beam search decoding
      decodes, scores, logprobs = decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        input_texts=text_encoder_inputs,
        **decoder_params)
    else:
      raise NotImplementedError

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
        return decodes[:, -1, :], dict(img=img, scores=scores[:, -1],
                                       logprobs=None if logprobs is None else logprobs[:, -1],
                                       img_tokens=image_decodes[:, -1])

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores, 'logprobs': logprobs}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1], 'logprobs': None if logprobs is None else logprobs[:, -1]}


class MultiImageEncoderDecoderPairModel(MultiImageEncoderDecoderModel):
  FEATURE_CONVERTER_CLS = UnifiedIOPairFeatureConverter

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
      mutable: flax_scope.CollectionFilter = False
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None

    return self.module.apply(
      {'params': params},
      batch['text_encoder_inputs'],
      batch['image_encoder_inputs'],
      batch['text_decoder_inputs_positive'],
      batch['image_decoder_targets'],
      batch['text_decoder_targets_positive'],
      text_encoder_masks=batch.get('text_encoder_masks', None),
      image_encoder_masks=batch.get('image_input_masks', None),
      image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
      text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
      text_decoder_masks=batch.get('text_decoder_masks_positive', None),
      image_decoder_masks=batch.get('image_target_masks', None),
      text_decoder_segment_ids=batch.get('text_decoder_segment_ids_positive', None),
      text_decoder_positions=batch.get('text_decoder_positions_positive', None),
      cache_text_length=self._text_decoder_length,
      cache_image_length=self._image_decoder_length,
      decode=False,
      enable_dropout=rngs is not None,
      rngs=rngs,
      mutable=mutable), \
           self.module.apply(
             {'params': params},
             batch['text_encoder_inputs'],
             batch['image_encoder_inputs'],
             batch['text_decoder_inputs_negative'],
             batch['image_decoder_targets'],
             batch['text_decoder_targets_negative'],
             text_encoder_masks=batch.get('text_encoder_masks', None),
             image_encoder_masks=batch.get('image_input_masks', None),
             image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
             text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
             text_decoder_masks=batch.get('text_decoder_masks_negative', None),
             image_decoder_masks=batch.get('image_target_masks', None),
             text_decoder_segment_ids=batch.get('text_decoder_segment_ids_negative', None),
             text_decoder_positions=batch.get('text_decoder_positions_negative', None),
             cache_text_length=self._text_decoder_length,
             cache_image_length=self._image_decoder_length,
             decode=False,
             enable_dropout=rngs is not None,
             rngs=rngs,
             mutable=mutable)

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
    label_smoothing = (
      self._label_smoothing if label_smoothing is None else label_smoothing)
    (text_logits_positive, image_logits, image_decoder_targets), \
    (text_logits_negative, _, _) = self._compute_logits(params, batch, dropout_rng)
    z_loss = 0.0

    text_weights_positive = batch.get('text_decoder_masks_positive', None)
    text_weights_negative = batch.get('text_decoder_masks_negative', None)
    image_weights = batch.get('image_target_loss_masks', None)

    text_loss_positive, text_z_loss, _ = losses.compute_weighted_cross_entropy(
      text_logits_positive,
      targets=batch['text_decoder_targets_positive'],
      weights=text_weights_positive,
      label_smoothing=label_smoothing,
      z_loss=0.0,
      loss_normalizing_factor=loss_normalizing_factor,
      loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
      vocab_size=33152,
      modality='text',
      return_sum=False,
    )
    text_z_loss = jnp.sum(text_z_loss)

    text_loss_negative, _, _ = losses.compute_weighted_cross_entropy(
      text_logits_negative,
      targets=batch['text_decoder_targets_negative'],
      weights=text_weights_negative,
      label_smoothing=label_smoothing,
      z_loss=0.0,
      loss_normalizing_factor=loss_normalizing_factor,
      loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
      vocab_size=33152,
      modality='text',
      return_sum=False,
    )

    score_positive = jnp.sum(jnp.exp(-text_loss_positive) * text_weights_positive, axis=1) / jnp.sum(text_weights_positive, axis=1)  # [batch]
    score_negative = jnp.sum(jnp.exp(-text_loss_negative) * text_weights_negative, axis=1) / jnp.sum(text_weights_negative, axis=1)  # [batch]
    pairwise_loss = -jnp.sum(jnp.log(jnp.exp(score_positive) / (jnp.exp(score_positive) + jnp.exp(score_negative))))

    text_loss_positive = jnp.sum(text_loss_positive)

    image_loss, image_z_loss, image_weight_sum = losses.compute_weighted_cross_entropy(
      image_logits,
      targets=image_decoder_targets,
      weights=image_weights,
      label_smoothing=label_smoothing,
      z_loss=z_loss,
      loss_normalizing_factor=loss_normalizing_factor,
      loss_normalizing_by_weight_sum=loss_normalizing_by_weight_sum,
      vocab_size=16384,
      modality='image')

    metrics = compute_base_metrics(
      logits=text_logits_positive,
      image_logits=image_logits,
      targets=batch['text_decoder_targets_positive'],
      image_targets=image_decoder_targets,
      mask=text_weights_positive,
      image_mask=image_weights,
      loss=text_loss_positive,
      image_loss=image_loss,
      z_loss=text_z_loss,
      image_z_loss=image_z_loss)

    loss = pairwise_loss + text_loss_positive + image_loss
    return loss, metrics

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
    text_decoder_shape = input_shapes['text_decoder_inputs_positive']
    text_decoder_type = input_types.get('text_decoder_inputs_positive', jnp.float32)
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
