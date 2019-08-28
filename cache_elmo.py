from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='cache elmo embedding')

    parser.add_argument('--dataset', type=str, default='vispro',
                        help='dataset: vispro, vispro_cdd, vispro_mscoco')
    
    args = parser.parse_args()
    return args

def build_elmo():
  token_ph = tf.placeholder(tf.string, [None, None])
  len_ph = tf.placeholder(tf.int32, [None])
  elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
  lm_embeddings = elmo_module(
      inputs={"tokens": token_ph, "sequence_len": len_ph},
      signature="tokens", as_dict=True)
  word_emb = lm_embeddings["word_emb"]
  lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                     lm_embeddings["lstm_outputs1"],
                     lm_embeddings["lstm_outputs2"]], -1)
  return token_ph, len_ph, lm_emb

def cache_dataset(data_path, session, dataset, token_ph, len_ph, lm_emb, out_file):
  with open(data_path) as in_file:
    for doc_num, line in enumerate(in_file.readlines()):
      example = json.loads(line)
      sentences = example["sentences"]
      
      if dataset == 'vispro':
        caption = sentences.pop(0)

      max_sentence_length = max(len(s) for s in sentences)
      tokens = [[""] * max_sentence_length for _ in sentences]
      text_len = np.array([len(s) for s in sentences])

      for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
          tokens[i][j] = word
      tokens = np.array(tokens)

      if dataset == 'vispro':
        # extract dialog
        tf_lm_emb_dial = session.run(lm_emb, feed_dict={
            token_ph: tokens,
            len_ph: text_len
        })
        file_key = example["doc_key"].replace("/", ":")
        group = out_file.create_group(file_key)
        for i, (e, l) in enumerate(zip(tf_lm_emb_dial, text_len)):
          e = e[:l, :, :]
          group[str(i + 1)] = e

        # extract caption alone 
        # extract spans from caption
        caption_NPs = example['correct_caption_NPs']
        file_key = file_key + ':cap'
        group = out_file.create_group(file_key)
        # caption_NPs might be empty
        if len(caption_NPs) == 0:
          continue
        # extract elmo feature for all spans
        span_len = [c[1] - c[0] + 1 for c in caption_NPs]
        span_list = [[""] * max(span_len) for _ in caption_NPs]
        for i, (span_start, span_end) in enumerate(caption_NPs):
          for j, index in enumerate(range(span_start, span_end + 1)):
            span_list[i][j] = caption[index].lower()
        span_list = np.array(span_list)
        tf_lm_emb_cap = session.run(lm_emb, feed_dict={
            token_ph: span_list,
            len_ph: span_len
        })
        for i, (e, l) in enumerate(zip(tf_lm_emb_cap, span_len)):
          e = e[:l, :, :]
          group[str(i)] = e

      else:
        tf_lm_emb = session.run(lm_emb, feed_dict={
            token_ph: tokens,
            len_ph: text_len
        })
        file_key = example["doc_key"].replace("/", ":")
        group = out_file.create_group(file_key)
        for i, (e, l) in enumerate(zip(tf_lm_emb, text_len)):
          e = e[:l, :, :]
          group[str(i)] = e

      if doc_num % 10 == 0:
        print(f"Cached {doc_num + 1} documents in {data_path}")

if __name__ == "__main__":
  token_ph, len_ph, lm_emb = build_elmo()

  args = parse_args()
  if args.dataset == 'vispro':
    json_filenames = ['data/' + s + '.vispro.1.1.jsonlines'
                      for s in ['train', 'val', 'test']]
  elif args.dataset == 'vispro_cdd':
    json_filenames = ['data/cdd_np.vispro.1.1.jsonlines']
  elif args.dataset == 'vispro_mscoco':
    json_filenames = ['data/mscoco_label.jsonlines']
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    h5_filename = "data/elmo_cache.%s.hdf5" % args.dataset
    out_file = h5py.File(h5_filename, "w")
    for json_filename in json_filenames:
      cache_dataset(json_filename, session, args.dataset, token_ph, len_ph, lm_emb, out_file)
    out_file.close()
