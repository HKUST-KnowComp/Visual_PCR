from __future__ import absolute_import
from __future__ import division

import json
import os
import os.path as osp
import argparse
import sys
import numpy as np

import tensorflow as tf
from model import VisCoref
import util

parser = argparse.ArgumentParser(description='predict coreference cluster on trained model')
parser.add_argument('model', type=str,
                    help='model name to evaluate')
parser.add_argument('--step', type=str, default='max',
                    help='global steps to restore from')
parser.add_argument('--split', type=str, default='test',
                    help='split to evaluate, test or val')
parser.add_argument('--input_dir', type=str, default='data',
                    help='input dir')
parser.add_argument('--output_dir', type=str, default='output',
                    help='output dir')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == "__main__":
  args = parser.parse_args()
  if len(sys.argv) == 1:
    sys.argv.append(args.model)
  else:
    sys.argv[1] = args.model
  config = util.initialize_from_env()
  input_filename = args.split + '.vispro.1.1.jsonlines'
  output_filename = args.split + '.vispro.1.1.prediction.jsonlines'
  input_filename = osp.join(args.input_dir, input_filename)
  output_filename = osp.join(args.output_dir, args.model, output_filename)

  model = VisCoref(config)

  # Create output dir
  output_dir = osp.split(output_filename)[0]
  if not osp.exists(output_dir):
    os.makedirs(output_dir)

  configtf = tf.ConfigProto()
  configtf.gpu_options.allow_growth = True
  with tf.Session(config=configtf) as session:
    model.restore(session, args.step)

    if config["use_im"]:
      predicted_att_grids = {}
    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}

          outputs = session.run(model.predictions, feed_dict=feed_dict)
          candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, tokens_cdd, tokens_obj, att_grid = outputs

          tokens_cdd_list = []
          for i in range(tokens_cdd.shape[0]):
            cdd_np = []
            for j in range(tokens_cdd.shape[1]):
              if tokens_cdd[i][j] != '':
                cdd_np.append(tokens_cdd[i][j])
            tokens_cdd_list.append(cdd_np)
          example["cdd_sentences"] = tokens_cdd_list

          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          if config["use_im"]:
            example["predicted_clusters"], predicted_att_grids[example["doc_key"]], _ = model.get_predicted_clusters_attention(top_span_starts, top_span_ends, att_grid, predicted_antecedents)
          else:
            example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

          output_file.write(json.dumps(example, cls=MyEncoder))
          output_file.write("\n")
          if example_num % 100 == 0:
            print(f"Decoded {example_num + 1} examples.")

    print(f"Output saved to {output_filename}")
    if config["use_im"]:
      output_filename = output_filename.replace('.jsonlines', '.att.npz')
      np.savez(output_filename, att=predicted_att_grids)
      print(f"Attention grids saved to {output_filename}")
