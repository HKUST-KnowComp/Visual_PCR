from __future__ import absolute_import
from __future__ import division

import json


def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for sentence in json.loads(line)["sentences"]:
          for word in sentence:
            vocab.update(word)
  vocab = sorted(list(vocab))
  with open(output_filename, "w") as f:
    for char in vocab:
      f.write(u"{}\n".format(char))
  print(f"Wrote {len(vocab)} characters to {output_filename}")

if __name__ == "__main__":
  json_filenames = ['data/' + s + '.vispro.1.1.jsonlines'
                      for s in ['train', 'val', 'test']]
  get_char_vocab(json_filenames, 'data/char_vocab.txt')
