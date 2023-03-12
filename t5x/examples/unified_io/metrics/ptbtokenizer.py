
#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import sys
import subprocess
import tempfile
import itertools

# path to the stanford corenlp jar
from os.path import expanduser

STANFORD_CORENLP_3_4_1_JAR = expanduser('~/stanford-corenlp-3.4.1.jar')

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
  """Python wrapper of Stanford PTBTokenizer"""

  def tokenize(self, captions_for_image):
    cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
           'edu.stanford.nlp.process.PTBTokenizer', \
           '-preserveLines', '-lowerCase']

    # ======================================================
    # prepare data for PTB Tokenizer
    # ======================================================
    final_tokenized_captions_for_image = {}
    image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
    sentences = [c.replace('\n', ' ') for k, v in captions_for_image.items() for c in v]
    _image_id, _sentences = [], []
    for img_id, sent in zip(image_id, sentences):
      if sent.strip() == "":
        if img_id not in final_tokenized_captions_for_image:
          final_tokenized_captions_for_image[img_id] = []
        final_tokenized_captions_for_image[img_id].append(sent)
      else:
        _image_id.append(img_id)
        _sentences.append(sent)
    image_id, sentences = _image_id, _sentences
    sentences = "\n".join(sentences)

    # ======================================================
    # tokenize sentence
    # ======================================================
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE,  stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip().encode("utf-8"))[0].decode("utf-8")
    lines = token_lines.split('\n')

    if p_tokenizer.returncode != 0:
      raise RuntimeError(f"Tokenizer returned error code {p_tokenizer.returncode}")

    # ======================================================
    # create dictionary for tokenized captions
    # ======================================================
    for k, line in zip(image_id, lines):
      if not k in final_tokenized_captions_for_image:
        final_tokenized_captions_for_image[k] = []
      tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                    if w not in PUNCTUATIONS])
      final_tokenized_captions_for_image[k].append(tokenized_caption)
    return final_tokenized_captions_for_image