from __future__ import absolute_import
from __future__ import division

import json
import argparse
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description='filter golve embedding')

    parser.add_argument('--embedding', type=str, default='glove.840B.300d.txt',
                        help='glove embedding file')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
  args = parse_args()

  json_filenames = ['data/' + s + '.vispro.1.1.jsonlines'
                      for s in ['train', 'val', 'test']]

  words_to_keep = set()
  for json_filename in json_filenames:
    print(f'Open {json_filename}')
    with open(json_filename) as json_file:
      for line in json_file.readlines():
        for sentence in json.loads(line)["sentences"]:
          words_to_keep.update(sentence)

  print(f"Found {len(words_to_keep)} words in {len(json_filenames)} dataset(s).")

  total_lines = 0
  kept_lines = 0
  out_filename = "{}.filtered".format(args.embedding)
  with open(osp.join('data', args.embedding)) as in_file:
    with open(osp.join('data', out_filename), "w") as out_file:
      for line in in_file.readlines():
        total_lines += 1
        word = line.split()[0]
        if word in words_to_keep:
          kept_lines += 1
          out_file.write(line)

  print(f"Kept {kept_lines} out of {total_lines} lines.")
  print(f"Wrote result to {out_filename}.")
