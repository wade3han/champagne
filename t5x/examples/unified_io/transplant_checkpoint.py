"""
To transplant image projection layers into cosmo (t5) models.
"""
import jax
import numpy as np
from flax import serialization
from tensorflow.io import gfile

cosmo_ckpt = 'cosmo_checkpoint'
uio_ckpt = 'uio_checkpoint'


def get_checkpoint_contents(ckpt_path):
  with gfile.GFile(ckpt_path, 'rb') as fp:
    # TODO(adarob): Use threaded reading as in flax.checkpoints.
    raw_contents = fp.read()
    if raw_contents.startswith(b'model_checkpoint_path'):
      raise ValueError(
        'Attempting to restore a TensorFlow checkpoint as a native T5X '
        'checkpoint. Use `restore_from_tf_checkpoint` instead. Path: ' +
        ckpt_path)

    # `ckpt_contents['optimizer']` is a pytree with a realized np.array for
    # leaves (params or states) written as msgpack and a ts.Spec (in a dict)
    # for leaves written by TensorStore.
    ckpt_contents = serialization.msgpack_restore(raw_contents)
    return ckpt_contents


cosmo_contents = get_checkpoint_contents(cosmo_ckpt)
uio_contents = get_checkpoint_contents(uio_ckpt)

# image projection layer from uio
image_proj_target = uio_contents['optimizer']['target']['encoder']['image_projection']
image_proj_state = uio_contents['optimizer']['state']['param_states']['encoder']['image_projection']
image_proj_target['kernel']['kvstore']['path'] = 'target.video_frames_encoder.image_projection.kernel/'

# position embedding from uio
image_pos_emb_target = uio_contents['optimizer']['target']['encoder']['position_embedding']
image_pos_emb_state = uio_contents['optimizer']['state']['param_states']['encoder']['position_embedding']
image_pos_emb_target['embedding']['kvstore']['path'] = 'target.video_frames_encoder.position_embedding.embedding/'

# transplant those things into cosmo contents
cosmo_contents['optimizer']['target']['video_frames_encoder'] = {'image_projection': image_proj_target,
                                                                 'position_embedding': image_pos_emb_target}
cosmo_contents['optimizer']['state']['param_states']['video_frames_encoder'] = {'image_projection': image_proj_state,
                                                                                'position_embedding': image_pos_emb_state}

assert cosmo_contents['optimizer']['target']['video_frames_encoder']['image_projection']['kernel']['kvstore'][
         'path'] == 'target.video_frames_encoder.image_projection.kernel/'
assert len(cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata'][
             'chunks']) == 2
assert \
cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata']['chunks'][
  0] == 832
assert \
cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata']['chunks'][
  1] == 2048


def fn(x):
  if isinstance(x, list):
    print(x)
    return np.array(x)
  else:
    return x


def walk_dict(d):
  for k, v in sorted(d.items(), key=lambda x: x[0]):
    if isinstance(v, dict):
      walk_dict(v)
    else:
      if k == 'chunks' or k == 'shape':
        d[k] = fn(v)

walk_dict(cosmo_contents)

with gfile.GFile('uio_transplanted_checkpoint', 'wb') as fp:
  fp.write(serialization.to_bytes(cosmo_contents))
print('Done!')

# check if it worked
transplanted_cosmo_contents = get_checkpoint_contents('uio_transplanted_checkpoint')
assert transplanted_cosmo_contents['optimizer']['target']['video_frames_encoder']['image_projection']['kernel']['kvstore']['path'] == 'target.video_frames_encoder.image_projection.kernel/'
assert len(transplanted_cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata']['chunks']) == 2
assert transplanted_cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata']['chunks'][0] == 832
assert transplanted_cosmo_contents['optimizer']['target']['video_frames_encoder']['position_embedding']['embedding']['metadata']['chunks'][1] == 2048
