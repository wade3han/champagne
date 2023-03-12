import tensorflow as tf
import seqio
from t5x.examples.unified_io.data.tasks import TaskRegistry
from t5x.examples.unified_io.data.mixtures import MixtureRegistry

from t5x.examples.unified_io.data.data_utils import UnifiedIOFeatureConverter, get_default_vocabulary

import aux_fns

print(tf.executing_eagerly())
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# tfds_data_dir = ''
# seqio.set_tfds_data_dir_override(tfds_data_dir)

dataset = seqio.get_mixture_or_task("class_specific_detection_coco_2017").get_dataset(
    sequence_length={"text_inputs": 256, "text_targets": 128, "is_training": True, "image_input_samples": 576},
    split="validation",
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42,
    shuffle=False,
)

converter = UnifiedIOFeatureConverter(pack=False, use_custom_packing_ops=False)
dataset = converter(dataset, {"text_inputs": 256, "text_targets": 128})
vocab = get_default_vocabulary()

print(dataset)
for _, ex in zip(range(10000), dataset.as_numpy_iterator()):
  print('Inputs')
  print(ex['text_encoder_inputs'])
  print('Targets')
  print(ex['text_decoder_targets'])
  print('Decoded inputs')
  print(vocab.decode(ex['text_encoder_inputs'].tolist()))
  print('Decoded decoder inputs')
  print(vocab.decode(ex['text_decoder_inputs'].tolist()))
  print('Decoded targets')
  print(vocab.decode(ex['text_decoder_targets'].tolist()))
  import pdb; pdb.set_trace()
  
  from torchvision.utils import draw_bounding_boxes, save_image
  import torch
  import numpy as np
  image = ex['image_encoder_inputs']
  image_plot = torch.tensor(np.array(image, dtype=np.float32), dtype=torch.float32)
  image_plot = image_plot.permute(2,0,1)
  save_image(image_plot, 'input_image.jpg')

  # image = ex['image_decoder_targets']
  # image_plot = torch.tensor(np.array(image, dtype=np.float32), dtype=torch.float32)
  # image_plot = image_plot.permute(2,0,1)
  # image_plot = (image_plot + 1) / 2
  # save_image(image_plot, 'target_image.jpg')
  # import pdb; pdb.set_trace()

#   vocab_size = 33100
#   text_sample_np = ex['text_decoder_targets']
#   BIN_START = vocab_size - 1000
#   # convert to x1y1x2y2 label format.
#   locations = []
#   labels = []
#   cur = 0
#   import matplotlib.pyplot as plt
#   from matplotlib import patches 
#   import numpy as np
#   # locations = np.array(locations)
#   while cur < text_sample_np.shape[0] and text_sample_np[cur] != 0:
#       if text_sample_np[cur] > BIN_START:
#           nxt = cur + 4
#           locations.append(vocab_size-text_sample_np[cur:nxt] - 100)
#           labels.append(text_sample_np[nxt])
#       else:
#           nxt = cur + 1
#       cur = nxt
  
#   locations = np.array(locations)
#   # text_sample_decode = vocab.decode(locations[0])
#   # text_sample_decode = str(text_sample_decode).split(' ')
#   # convert to image size.
#   locations[:,0] = locations[:,0] / (1000-1) * 256
#   locations[:,2] = locations[:,2] / (1000-1) * 256 
#   locations[:,1] = locations[:,1] / (1000-1) * 256 
#   locations[:,3] = locations[:,3] / (1000-1) * 256

#   import matplotlib.pyplot as plt
#   from matplotlib import patches 
#   fig, ax = plt.subplots()
#   plt.axis('off')
#   # # Display the image
#   i = 0
#   x1,y1,x2,y2 = locations[i].tolist()
#   w1 = x2 - x1
#   h1 = y2 - y1
#   rect = patches.Rectangle((y1, x1), h1, w1, linewidth=1, edgecolor='r', facecolor='none')
#   ax.add_patch(rect)
#   ax.imshow(ex['image_decoder_targets'])
#   plt.savefig('foo.png', bbox_inches='tight')

#   import pdb; pdb.set_trace()