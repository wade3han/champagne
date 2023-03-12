# prepare vqa submission 

import json
import re

from settings import OWNER_NAME


def find_id(text):
  text = text.split(' ')
  out = []
  obj_out = []
  c = 0
  for t in text:
    if t.isdigit():
      if c < 10:
        out.append('#' + str(c) + '#')
        obj_out.append(int(t))
        c += 1
    else:
      out.append(t)
  return ' '.join(out), obj_out


ans_file_path = f'/home/{OWNER_NAME}/dialogue-toolbox/before.json'
ans_file_before = json.load(open(ans_file_path, 'r'))
for img in ans_file_before: img['inference_relation'] = 'before'

ans_file_path = f'/home/{OWNER_NAME}/dialogue-toolbox/intent.json'
ans_file_intent = json.load(open(ans_file_path, 'r'))
for img in ans_file_intent: img['inference_relation'] = 'intent'

ans_file_path = f'/home/{OWNER_NAME}/dialogue-toolbox/after.json'
ans_file_after = json.load(open(ans_file_path, 'r'))
for img in ans_file_after: img['inference_relation'] = 'after'

ans_file = ans_file_before + ans_file_intent + ans_file_after

anno_file_path = f'/home/{OWNER_NAME}/visual-comet/data/visualcomet/test_annots.json'
anno_file = json.load(open(anno_file_path, 'r'))

# add event index
for i, img in enumerate(anno_file):
  img['event_idx'] = i

anno_file_dict = {img['img_fn']: img for img in anno_file}

for img in anno_file:
  # img['img_fn']
  event, event_bbox_id = find_id(img['event'])
  k = img['img_fn'] + '-###-' + event
  anno_file_dict[k] = img

out = []
for img in ans_file:
  img_fn = img['example_id']
  words = [re.sub(r'<extra_id_(.*?)> ', r'', a.lower()) for a in [img["words"]]]

  anno = anno_file_dict[img_fn]
  example = {}
  example['img_fn'] = anno['img_fn']
  example['movie'] = anno['movie']
  example['metadata_fn'] = anno['metadata_fn']
  example['split'] = 'test'
  example['place'] = anno['place']
  example['event'] = anno['event']
  example['inference_relation'] = img['inference_relation']
  example['event_idx'] = anno['event_idx']
  example['generations'] = words
  out.append(example)
json.dump(out, open('scripts/eval/viscomet-val.json', 'w'))
