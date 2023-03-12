import jax
import jax.numpy as jnp
from flax import linen as nn

from t5x.examples.unified_io import layers
from t5x.examples.unified_io.layers import default_embed_init
from t5x.examples.unified_io.network import T5Config, VAEConfig, DiscreteVAE, Decoder, Initializer, EncoderLayer


class VideoEncoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config
  shared_embedding: nn.Module
  embedding_init: Initializer = default_embed_init

  def setup(self):
    cfg = self.config
    self.segment_embedding = layers.Embed(
      num_embeddings=cfg.num_seg_emb,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='segment_embedding')

    self.positon_embedding = layers.Embed(
      num_embeddings=cfg.encoder_max_text_length + cfg.encoder_max_image_length,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='position_embedding')

    self.image_position_embedding = layers.Embed(
      num_embeddings=8 if cfg.emb_dim == 2048 else 4,  # use 8 only for xl
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='image_position_embedding')

  def get_video_position_embedding(self, num_turns):
    if num_turns > 3:  # 7 --> num_interpolated_embeds = 2,
      video_position_embedding = self.image_position_embedding(jnp.arange(3, dtype=jnp.int32)[None])[:, :, None, :]
      n_interpolated_embeds = (num_turns - 3) // 2
      first_embeds = [
        video_position_embedding[:, 0:1, :, :] * (n_interpolated_embeds + 1 - i) / float(n_interpolated_embeds + 1) + \
        video_position_embedding[:, 1:2, :, :] * i / float(n_interpolated_embeds + 1)
        for i in range(1, n_interpolated_embeds + 1)
      ]
      second_embeds = [
        video_position_embedding[:, 1:2, :, :] * (n_interpolated_embeds + 1 - i) / float(n_interpolated_embeds + 1) + \
        video_position_embedding[:, 2:3, :, :] * i / float(n_interpolated_embeds + 1)
        for i in range(1, n_interpolated_embeds + 1)
      ]
      video_position_embedding = jnp.concatenate(
        [video_position_embedding[:, 0:1, :, :],
         *first_embeds,
         video_position_embedding[:, 1:2, :, :],
         *second_embeds,
         video_position_embedding[:, 2:3, :, :]], axis=1)
    else:
      video_position_embedding = self.image_position_embedding(
        jnp.arange(num_turns, dtype=jnp.int32)[None])[:, :, None, :]
    return video_position_embedding

  @nn.compact
  def __call__(self,
               text_encoder_inputs,
               image_encoder_inputs,
               txt_position_ids,
               img_position_ids,
               encoder_masks=None,
               deterministic=False):
    cfg = self.config
    assert text_encoder_inputs.ndim == 2  # [batch, length]
    assert image_encoder_inputs.ndim == 4  # [batch, num_image_frames, length, dim]
    h, w = cfg.default_image_size

    rel_emb = layers.RelativePositionBiases(
      num_buckets=32,
      img_num_buckets=8,
      max_distance=128,
      img_max_distance=20,
      num_heads=cfg.num_heads,
      img_width=w // cfg.image_patch_size,
      img_height=h // cfg.image_patch_size,
      dtype=cfg.dtype,
      embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                      'uniform'),
      name='relpos_bias')

    img_emb = image_encoder_inputs

    txt_pos_emb = self.positon_embedding(txt_position_ids)
    img_pos_emb = self.positon_embedding(img_position_ids + cfg.encoder_max_text_length)

    # img_emb: [B, L, D]
    img_emb = layers.DenseGeneral(
      cfg.emb_dim,
      dtype=cfg.dtype,
      kernel_axes=('image_patch', 'embed'),
      name='image_projection',
    )(img_emb)  # [batch, num_image_frames, length, dim]

    # do the text encoding
    # [batch, length] -> [batch, length, emb_dim]
    txt_emb = self.shared_embedding(text_encoder_inputs.astype('int32'))

    txt_segments = jnp.zeros(txt_emb.shape[1], dtype=jnp.int32)[None, ...]
    img_segments = jnp.ones(img_emb.shape[2], dtype=jnp.int32)[None, ...]

    txt_emb += self.segment_embedding(txt_segments)
    img_emb += self.segment_embedding(img_segments)[:, None, :, :]

    txt_emb += txt_pos_emb
    img_emb += img_pos_emb[:, None, :, :]  # [B, num_image_frames, L, D]

    # add image positional embedding to give informations
    ### perceiver style
    batch_size, num_turns, img_length = img_emb.shape[:3]
    video_position_embedding = self.get_video_position_embedding(num_turns)
    img_emb += video_position_embedding
    img_emb = jnp.mean(img_emb, axis=1)

    txt_emb = layers.LayerNorm(
      dtype=cfg.dtype, name='txt_emb_pre_ln')(txt_emb)

    img_emb = layers.LayerNorm(
      dtype=cfg.dtype, name='img_emb_pre_ln')(img_emb)

    position_embedding = jnp.concatenate([txt_pos_emb, img_pos_emb], axis=1)

    position_embedding = layers.LayerNorm(
      dtype=cfg.dtype, name='pe_pre_ln')(position_embedding)

    # get absolute position bias.
    pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_q_linear',
    )(position_embedding)

    pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_k_linear',
    )(position_embedding)

    pos_scaling = float(cfg.emb_dim / cfg.num_heads) ** -0.5
    abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', pos_q, pos_k) * pos_scaling

    x = jnp.concatenate([txt_emb, img_emb], axis=1)
    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
        config=cfg, relative_embedding=rel_emb,
        name=f'layers_{lyr}')(x, txt_position_ids, img_position_ids, abs_pos_bias, encoder_masks, deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic), position_embedding


class VideoTransformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config
  vae_config: VAEConfig

  def setup(self):
    cfg = self.config
    vae_config = self.vae_config

    self.shared_embedding = layers.Embed(
      num_embeddings=cfg.vocab_size + cfg.image_vocab_size,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='token_embedder')

    self.discrete_vae = DiscreteVAE(config=vae_config)
    self.encoder = VideoEncoder(
      config=cfg,
      shared_embedding=self.shared_embedding,
    )
    self.decoder = Decoder(
      config=cfg,
      shared_embedding=self.shared_embedding)

    total_vocab_size = cfg.vocab_size + cfg.image_vocab_size
    self.logit_range = jnp.reshape(jnp.arange(total_vocab_size), [1, 1, -1])
    self.image_logits_mask = jnp.reshape(self.logit_range < cfg.vocab_size, [1, -1])
    self.text_logits_mask = jnp.reshape(self.logit_range >= cfg.vocab_size, [1, -1])

  def encode(self,
             text_encoder_inputs,
             image_encoder_inputs,
             text_encoder_masks,
             image_encoder_masks,
             image_encoder_pos_ids,
             text_encoder_pos_ids,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert text_encoder_inputs.ndim == 2  # (batch, len)
    bs = text_encoder_inputs.shape[0]

    if text_encoder_masks is None:
      text_encoder_masks = text_encoder_inputs > 0

    image_length = image_encoder_inputs.shape[2]

    if image_encoder_masks is None:
      image_encoder_masks = jnp.ones([image_encoder_inputs.shape[0], image_length], dtype=jnp.bool_)

    if image_encoder_pos_ids is None:
      image_encoder_pos_ids = jnp.arange(image_length, dtype=jnp.int32)
      image_encoder_pos_ids = jnp.expand_dims(image_encoder_pos_ids, axis=0)
      image_encoder_pos_ids = jnp.tile(image_encoder_pos_ids, [bs, 1])

    if text_encoder_pos_ids is None:
      text_encoder_pos_ids = jnp.arange(text_encoder_inputs.shape[1], dtype=jnp.int32)
      text_encoder_pos_ids = jnp.expand_dims(text_encoder_pos_ids, axis=0)
      text_encoder_pos_ids = jnp.tile(text_encoder_pos_ids, [bs, 1])

    encoder_masks = jnp.concatenate([text_encoder_masks, image_encoder_masks], axis=1)
    encoder_attn_masks = layers.make_attention_mask(
      encoder_masks, encoder_masks, dtype=cfg.dtype)

    return self.encoder(
      text_encoder_inputs,
      image_encoder_inputs,
      text_encoder_pos_ids,
      image_encoder_pos_ids,
      encoder_masks=encoder_attn_masks,
      deterministic=not enable_dropout
    ), encoder_masks

  def decode(
      self,
      encoded,
      encoder_masks,
      text_decoder_inputs,
      image_decoder_inputs,
      text_decoder_targets,
      image_decoder_targets,
      text_decoder_masks=None,
      image_decoder_masks=None,
      text_decoder_segment_ids=None,
      text_decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    if text_decoder_masks is None:
      text_decoder_masks = text_decoder_targets > 0

    if image_decoder_masks is None:
      image_decoder_masks = jnp.ones(image_decoder_inputs.shape, dtype=jnp.bool_)

    if text_decoder_segment_ids is not None:
      decoder_segment_ids = jnp.concatenate([text_decoder_segment_ids, jnp.ones(image_decoder_masks.shape)], axis=1)
    else:
      decoder_segment_ids = None

    decoder_masks = jnp.concatenate([text_decoder_masks, image_decoder_masks], axis=1)
    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=decoder_masks,
      dtype=cfg.dtype,
      decoder_segment_ids=decoder_segment_ids)

    encoder_decoder_mask = layers.make_attention_mask(
      decoder_masks, encoder_masks, dtype=cfg.dtype)

    decoder_inputs = jnp.concatenate([text_decoder_inputs, image_decoder_inputs], axis=1)

    if text_decoder_positions is None:
      text_decoder_positions = jnp.arange(text_decoder_inputs.shape[1], dtype=jnp.int32)[None, ...]
      image_decoder_positions = jnp.arange(image_decoder_inputs.shape[1], dtype=jnp.int32)[None, ...]
    else:
      image_decoder_positions = jnp.arange(image_decoder_inputs.shape[1], dtype=jnp.int32)[None, ...]
      image_decoder_positions = jnp.tile(image_decoder_positions, [image_decoder_inputs.shape[0], 1])

    decoder_positions = jnp.concatenate([
      text_decoder_positions,
      cfg.decoder_max_text_length + image_decoder_positions],
      axis=1)

    decoder_segments = jnp.expand_dims(
      jnp.concatenate([
        jnp.zeros(text_decoder_inputs.shape[1], dtype=jnp.int32),
        jnp.ones(image_decoder_inputs.shape[1], dtype=jnp.int32)],
        axis=0),
      axis=0)

    logits = self.decoder(
      encoded,
      decoder_positions=decoder_positions,
      decoder_segments=decoder_segments,
      decoder_inputs=decoder_inputs,
      decoder_attn_mask=decoder_attn_mask,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      image_decoder_positions=image_decoder_positions,
      text_decoder_positions=text_decoder_positions)

    # mask the logits.
    text_length = text_decoder_inputs.shape[1]
    seq_range = jnp.reshape(jnp.arange(logits.shape[1]), [1, -1, 1])
    logits_mask = (((seq_range >= text_length) & (self.logit_range < cfg.vocab_size)) |
                   (seq_range < text_length) & (self.logit_range >= cfg.vocab_size))
    logits = jnp.where(logits_mask, -1e10, logits)
    text_logits = logits[:, :text_length]
    image_logits = logits[:, text_length:]

    return text_logits, image_logits, image_decoder_targets

  def decode_code(self, code_b):
    return self.discrete_vae.decode_code(code_b)

  def sample(
      self,
      encoded,
      encoder_masks,
      decoder_inputs,
      decoder_masks=None,
      decoder_segments=None,
      enable_dropout=True,
      decode=False,
      cur_index=None,
      image_decode_length=None,
      text_decode_length=None):

    cfg = self.config
    encoder_decoder_mask = layers.make_attention_mask(
      jnp.ones_like(decoder_inputs),
      encoder_masks,
      dtype=cfg.dtype)

    if decoder_masks is not None:
      decoder_attn_mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_masks,
        dtype=cfg.dtype)
    else:
      decoder_attn_mask = None

    image_decoder_positions = jnp.arange(image_decode_length)[None, ...]
    text_decoder_positions = jnp.arange(text_decode_length)[None, ...]

    decoder_positions = jnp.concatenate([
      text_decoder_positions,
      cfg.decoder_max_text_length + image_decoder_positions],
      axis=1)

    logits = self.decoder(
      encoded,
      decoder_inputs=decoder_inputs,
      decoder_positions=decoder_positions,
      decoder_segments=decoder_segments,
      decoder_attn_mask=decoder_attn_mask,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      image_decoder_positions=image_decoder_positions,
      text_decoder_positions=text_decoder_positions,
      cur_index=cur_index)

    return logits

  def __call__(self,
               text_encoder_inputs,
               image_encoder_inputs,
               text_decoder_inputs,
               image_decoder_targets,
               text_decoder_targets,
               text_encoder_masks=None,
               image_encoder_masks=None,
               text_decoder_masks=None,
               image_decoder_masks=None,
               image_encoder_pos_ids=None,
               text_encoder_pos_ids=None,
               text_decoder_segment_ids=None,
               text_decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               cache_text_length=None,
               cache_image_length=None,
               vae_decode: bool = False,
               return_targets=False
               ):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    cfg = self.config

    image_decoder_tokens = self.discrete_vae.get_codebook_indices(image_decoder_targets,
                                                                  vae_decode)  # 0 is the start token.
    # stop gradient.
    image_decoder_tokens = image_decoder_tokens + cfg.vocab_size
    image_decoder_tokens = jax.lax.stop_gradient(image_decoder_tokens)

    image_decoder_inputs = jnp.concatenate([
      jnp.zeros((image_decoder_tokens.shape[0], 1), dtype=jnp.int32) + cfg.vocab_size - 1,
      image_decoder_tokens[:, :-1]], axis=1)

    encoded, encoder_masks = self.encode(
      text_encoder_inputs,
      image_encoder_inputs,
      text_encoder_masks,
      image_encoder_masks,
      image_encoder_pos_ids,
      text_encoder_pos_ids,
      enable_dropout=enable_dropout)

    if cache_image_length is not None:
      image_decoder_inputs = image_decoder_inputs[:, :cache_image_length]
      image_decoder_tokens = image_decoder_tokens[:, :cache_image_length]
      if image_decoder_masks is not None:
        image_decoder_masks = image_decoder_masks[:, :cache_image_length]

    if cache_text_length is not None:
      text_decoder_inputs = text_decoder_inputs[:, :cache_text_length]
      text_decoder_targets = text_decoder_targets[:, :cache_text_length]
      if text_decoder_masks is not None:
        text_decoder_masks = text_decoder_masks[:, :cache_text_length]

    logits = self.decode(
      encoded,
      encoder_masks,
      text_decoder_inputs,
      image_decoder_inputs,
      text_decoder_targets,
      image_decoder_tokens,
      text_decoder_masks=text_decoder_masks,
      image_decoder_masks=image_decoder_masks,
      text_decoder_segment_ids=text_decoder_segment_ids,
      text_decoder_positions=text_decoder_positions,
      enable_dropout=enable_dropout,
      decode=decode)

    if return_targets:
      return logits
    else:
      return logits
