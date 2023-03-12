import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

import tensorflow as tf
import pdb

jax_model = nn.Conv(
  features=4, 
  kernel_size=(3,3),
  strides=(1, 1),
  padding='valid',
  )

key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (1,5,5,3)) # Dummy input
params = jax_model.init(key2, x) # Initialization call
params = dict(params)

tf_model = tf.keras.layers.Conv2D(
                filters=4,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                kernel_initializer='he_uniform',
                bias_initializer='zeros',
                name='conv_in')

tf_x = tf.constant(x)
_ = tf_model(tf_x)
tf_model.set_weights([params['params']['kernel'], params['params']['bias']])

pdb.set_trace()
print(tf.math.reduce_sum(tf_model(tf_x)))
print(jax_model.apply(params, x).sum())


tf_x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(tf_x)
print(tf.math.reduce_sum(tf_model(tf_x)))