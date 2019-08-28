from __future__ import absolute_import
from __future__ import division

import os
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import random

import util
import coref_ops
import metrics

class VisCoref(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]

    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]

    self.use_im = self.config["use_im"]
    im_obj_labels = [json.loads(line) for line in open(self.config["im_obj_label_path"], "r")]
    self.id2cat = {int(d["doc_key"]):d["sentences"][0] for d in im_obj_labels}
    if self.use_im:
      self.lm_obj_file = h5py.File(self.config["lm_obj_path"], "r")
      self.im_emb_size = self.config["im_emb_size"]

    self.vis_weight = self.config["vis_weight"]
    self.num_cdd_pool = self.config["num_cdd_pool"]
    self.lm_cdd_file = h5py.File(self.config["lm_cdd_path"], "r")
    with open(self.config["cdd_path"]) as f:
      self.cdd_nps = [json.loads(jsonline) for jsonline in f.readlines()]

    self.eval_data = None # Load eval data lazily.
    self.lm_file = h5py.File(self.config["lm_path"], "r")
    print(f'Loading elmo cache from {self.config["lm_path"]}')

    input_props = []
    input_props.append((tf.string, [None, None])) # Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings for cap.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings for dial.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None])) # Speaker IDs.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    input_props.append((tf.int32, [None])) # caption candidate starts.
    input_props.append((tf.int32, [None])) # caption candidate ends.
    input_props.append((tf.int32, [None])) # Text lengths cdd.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings cdd.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings cdd.
    input_props.append((tf.int32, [None, None, None])) # Character indices cdd.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings for cdd.
    input_props.append((tf.string, [None, None])) # Tokens cdd.
    input_props.append((tf.int32, [None])) # Text lengths obj.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings obj.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings obj.
    input_props.append((tf.int32, [None, None, None])) # Character indices obj.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings for obj.
    input_props.append((tf.string, [None, None])) # Tokens obj.      
    input_props.append((tf.bool, [])) # Has object.

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    self.max_eval_f1 = tf.Variable(0.0, name="max_eval_f1", trainable=False)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        global_step = session.run(self.global_step)
        random.seed(self.config["random_seed"] + global_step)
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session, step='max'):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    if step == 'max':
      path = "model.max.ckpt"
    else:
      path = "model-" + step
    checkpoint_path = os.path.join(self.config["log_dir"], path)
    print(f"Restoring from {checkpoint_path}")
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")

    group_cap = self.lm_file[file_key + ':cap']
    num_candidates = len(list(group_cap.keys()))
    candidates = [group_cap[str(i)][...] for i in range(num_candidates)]
    if len(candidates) > 0:
      lm_emb_cap = np.zeros([len(candidates), max(c.shape[0] for c in candidates), self.lm_size, self.lm_layers])
      for i, c in enumerate(candidates):
        lm_emb_cap[i, :c.shape[0], :, :] = c
    else:
      # to avoid empty lm_emb_cap
      lm_emb_cap = np.zeros([1, 1, self.lm_size, self.lm_layers])
      
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(1, num_sentences + 1)]
    lm_emb_dial = np.zeros([len(sentences), max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb_dial[i, :s.shape[0], :, :] = s

    return [lm_emb_cap, lm_emb_dial]

  def load_lm_embeddings_cdd(self, examples):
    candidates = [self.lm_cdd_file[e['doc_key']]['0'][...] for e in examples]
    lm_emb_cdd = np.zeros([len(candidates), max(c.shape[0] for c in candidates), self.lm_size, self.lm_layers])
    for i, c in enumerate(candidates):
      lm_emb_cdd[i, :c.shape[0], :, :] = c

    return lm_emb_cdd

  def load_lm_embeddings_obj(self, objs):
    objs = [self.lm_obj_file[str(obj)]['0'][...] for obj in objs]
    lm_emb_objs = np.zeros([len(objs), max(c.shape[0] for c in objs), self.lm_size, self.lm_layers])
    for i, c in enumerate(objs):
      lm_emb_objs[i, :c.shape[0], :, :] = c

    return lm_emb_objs

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]

    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        if i == 0:
          word = word.lower()
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    tokens = np.array(tokens)

    if self.num_cdd_pool > 0:
      # random pick samples to a candidate pool of fixed number
      num_cdd_pick = self.num_cdd_pool - len(example["correct_caption_NPs"])
      num_cdd_pick = max(1, num_cdd_pick)
      cdd_examples = []
      all_sentences = list()
      for sent in sentences:
        all_sentences += sent
      candidate_cur = example["pronoun_info"][-1]["candidate_NPs"]
      candidate_cur = [' '.join(all_sentences[c[0]:c[1]+1]) for c in candidate_cur]
      if not is_training:
        sample_times = 0
      while len(cdd_examples) < num_cdd_pick:
        if not is_training:
          random.seed(example["doc_key"] + str(sample_times))
          cdd_cur = random.choice(self.cdd_nps)
          sample_times += 1
        else:
          cdd_cur = random.choice(self.cdd_nps)
        # samples in candidate pools should not be the same as candidate nps
        cdd_text = ' '.join(cdd_cur["sentences"][0]).lower()
        repeat_flag = False
        for cdd in candidate_cur:
          if cdd.lower() == cdd_text:
            repeat_flag = True
            break
        if not repeat_flag:
          cdd_examples.append(cdd_cur)

      sentences_cdd = [s["sentences"][0] for s in cdd_examples]
      max_sentence_length_cdd = max(len(s) for s in sentences_cdd)
      max_word_length_cdd = max(max(max(len(w) for w in s) for s in sentences_cdd), max(self.config["filter_widths"]))
      text_len_cdd = np.array([len(s) for s in sentences_cdd])
      context_word_emb_cdd = np.zeros([len(sentences_cdd), max_sentence_length_cdd, self.context_embeddings.size])
      head_word_emb_cdd= np.zeros([len(sentences_cdd), max_sentence_length_cdd, self.head_embeddings.size])
      char_index_cdd = np.zeros([len(sentences_cdd), max_sentence_length_cdd, max_word_length_cdd])
      tokens_cdd = [[""] * max_sentence_length_cdd for _ in sentences_cdd]
      for i, sentence_cdd in enumerate(sentences_cdd):
        for j, word_cdd in enumerate(sentence_cdd):
          tokens_cdd[i][j] = word_cdd
          context_word_emb_cdd[i, j] = self.context_embeddings[word_cdd]
          head_word_emb_cdd[i, j] = self.head_embeddings[word_cdd]
          char_index_cdd[i, j, :len(word_cdd)] = [self.char_dict[c] for c in word_cdd]
      tokens_cdd = np.array(tokens_cdd)

      lm_emb_cdd = self.load_lm_embeddings_cdd(cdd_examples)

      for len_cdd in text_len_cdd:
        example["speakers"].append(['caption',] * len_cdd)

    if self.use_im:
      detections = example["object_detection"]
      has_obj = len(detections) > 0
      detections.append(0)
      sentences_obj = [self.id2cat[i] for i in detections]
      max_sentence_length_obj = max(len(s) for s in sentences_obj)
      max_word_length_obj = max(max(max(len(w) for w in s) for s in sentences_obj), max(self.config["filter_widths"]))
      text_len_obj = np.array([len(s) for s in sentences_obj])
      context_word_emb_obj = np.zeros([len(sentences_obj), max_sentence_length_obj, self.context_embeddings.size])
      head_word_emb_obj= np.zeros([len(sentences_obj), max_sentence_length_obj, self.head_embeddings.size])
      char_index_obj = np.zeros([len(sentences_obj), max_sentence_length_obj, max_word_length_obj])
      tokens_obj = [[""] * max_sentence_length_obj for _ in sentences_obj]
      for i, sentence_obj in enumerate(sentences_obj):
        for j, word_obj in enumerate(sentence_obj):
          tokens_obj[i][j] = word_obj
          context_word_emb_obj[i, j] = self.context_embeddings[word_obj]
          head_word_emb_obj[i, j] = self.head_embeddings[word_obj]
          char_index_obj[i, j, :len(word_obj)] = [self.char_dict[c] for c in word_obj]
      lm_emb_obj = self.load_lm_embeddings_obj(example["object_detection"])

    speakers = util.flatten(example["speakers"])
    speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    doc_key = example["doc_key"]
    caption_candidates = example["correct_caption_NPs"]
    if len(caption_candidates) == 0:
      # add 1 NP to avoid empty candidates
      caption_candidates = [[0, 0]]
    candidate_starts_caption, candidate_ends_caption = self.tensorize_mentions(caption_candidates)

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    lm_emb_cap, lm_emb_dial = self.load_lm_embeddings(doc_key)

    example_tensors = [tokens, context_word_emb, head_word_emb, lm_emb_cap, lm_emb_dial, char_index, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids, candidate_starts_caption, candidate_ends_caption]
    example_tensors.extend([text_len_cdd, context_word_emb_cdd, head_word_emb_cdd, char_index_cdd, lm_emb_cdd, tokens_cdd])
    if self.use_im:
      example_tensors.extend([text_len_obj, context_word_emb_obj, head_word_emb_obj, char_index_obj, lm_emb_obj, tokens_obj, has_obj])
    else:
      example_tensors.extend([[0], np.zeros([0, 0, self.context_embeddings.size]),
                             np.zeros([0, 0, self.head_embeddings.size]),
                             np.zeros([0, 0, 1]), np.zeros([0, 0, self.lm_size, self.lm_layers]),
                             np.zeros([0, 1]), False])

      
    return example_tensors

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c, top_span_cdd_pool_flag=None):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]

    num_cdd_in_pool = tf.reduce_sum(tf.cast(top_span_cdd_pool_flag, tf.int32))
    num_cdd_in_dial = k - num_cdd_in_pool
    top_span_range_cdd = tf.concat([tf.zeros(num_cdd_in_pool, tf.int32), tf.range(1, num_cdd_in_dial + 1)], 0)
    antecedent_offsets = tf.expand_dims(top_span_range_cdd, 1) - tf.expand_dims(top_span_range_cdd, 0) # [k, k]

    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask)) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb_cap, lm_emb_dial, char_index, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids, candidate_starts_caption, candidate_ends_caption, text_len_cdd, context_word_emb_cdd, head_word_emb_cdd, char_index_cdd, lm_emb_cdd, tokens_cdd, text_len_obj, context_word_emb_obj, head_word_emb_obj, char_index_obj, lm_emb_obj, tokens_obj, has_obj):
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    # for all sentences including caption
    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    # get char embedding by conv1d on char embeddings of each word
    if self.config["char_embedding_size"] > 0:
      char_emb_all = tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]])
      char_emb = tf.gather(char_emb_all, char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    # for candidate pool
    num_sentences_cdd = tf.shape(context_word_emb_cdd)[0]
    max_sentence_length_cdd = tf.shape(context_word_emb_cdd)[1]

    context_emb_list_cdd = [context_word_emb_cdd]
    head_emb_list_cdd = [head_word_emb_cdd]

    # get char embedding by conv1d on char embeddings of each word
    if self.config["char_embedding_size"] > 0:
      char_emb_cdd = tf.gather(char_emb_all, char_index_cdd) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb_cdd = tf.reshape(char_emb_cdd, [num_sentences_cdd * max_sentence_length_cdd, util.shape(char_emb_cdd, 2), util.shape(char_emb_cdd, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb_cdd = util.cnn(flattened_char_emb_cdd, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb_cdd = tf.reshape(flattened_aggregated_char_emb_cdd, [num_sentences_cdd, max_sentence_length_cdd, util.shape(flattened_aggregated_char_emb_cdd, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list_cdd.append(aggregated_char_emb_cdd)
      head_emb_list_cdd.append(aggregated_char_emb_cdd)

    context_emb_cdd = tf.concat(context_emb_list_cdd, 2) # [num_sentences, max_sentence_length, emb]
    head_emb_cdd = tf.concat(head_emb_list_cdd, 2) # [num_sentences, max_sentence_length, emb]

    # extract embedding for NPs in caption here
    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    text_len_cap = candidate_ends_caption - candidate_starts_caption + 1
    max_span_width_cap = tf.math.reduce_max(text_len_cap)
    span_indices_cap = tf.expand_dims(tf.range(max_span_width_cap), 0) + tf.expand_dims(candidate_starts_caption, 1) # [num_candidates_cap, max_span_width_cap]
    span_indices_cap = tf.minimum(text_len[0] - 1, span_indices_cap) # [num_candidates_cap, max_span_width_cap]
    context_emb_cap = tf.gather(context_emb[0], span_indices_cap) # [num_candidates_cap, max_span_width_cap, emb]
    head_emb_cap = tf.gather(head_emb[0], span_indices_cap) # [num_candidates_cap, max_span_width_cap, emb]

    # project lm_num_layer to 1 and scale
    lm_emb_size = util.shape(lm_emb_dial, 2)
    lm_num_layers = util.shape(lm_emb_dial, 3)
    # for sentences in dialog only
    num_sentences_dial = util.shape(lm_emb_dial, 0)
    max_sentence_length_dial = util.shape(lm_emb_dial, 1)
    # for caption
    num_candidates_cap = util.shape(lm_emb_cap, 0)
    max_candidate_length_cap = util.shape(lm_emb_cap, 1)
    # get projection and scaling parameter
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
    # for lm emb of cap
    flattened_lm_emb_cap = tf.reshape(lm_emb_cap, [num_candidates_cap * max_candidate_length_cap * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb_cap = tf.matmul(flattened_lm_emb_cap, tf.expand_dims(self.lm_weights, 1)) # [num_candidates_cap * max_candidate_length_cap * emb, 1]
    aggregated_lm_emb_cap = tf.reshape(flattened_aggregated_lm_emb_cap, [num_candidates_cap, max_candidate_length_cap, lm_emb_size])
    aggregated_lm_emb_cap *= self.lm_scaling
    # for lm emb of dial
    flattened_lm_emb_dial = tf.reshape(lm_emb_dial, [num_sentences_dial * max_sentence_length_dial * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb_dial = tf.matmul(flattened_lm_emb_dial, tf.expand_dims(self.lm_weights, 1)) # [num_sentences_dial * max_sentence_length_dial * emb, 1]
    aggregated_lm_emb_dial = tf.reshape(flattened_aggregated_lm_emb_dial, [num_sentences_dial, max_sentence_length_dial, lm_emb_size])
    aggregated_lm_emb_dial *= self.lm_scaling
    # for lm emb of cdd
    num_candidates_cdd = util.shape(lm_emb_cdd, 0)
    max_candidate_length_cdd = util.shape(lm_emb_cdd, 1)
    flattened_lm_emb_cdd = tf.reshape(lm_emb_cdd, [num_candidates_cdd * max_candidate_length_cdd * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb_cdd = tf.matmul(flattened_lm_emb_cdd, tf.expand_dims(self.lm_weights, 1)) # [num_candidates_cdd * max_candidate_length_cdd * emb, 1]
    aggregated_lm_emb_cdd = tf.reshape(flattened_aggregated_lm_emb_cdd, [num_candidates_cdd, max_candidate_length_cdd, lm_emb_size])
    aggregated_lm_emb_cdd *= self.lm_scaling

    context_emb_dial = tf.concat([context_emb[1:, :max_sentence_length_dial], aggregated_lm_emb_dial], 2) # [num_sentences_dial, max_sentence_length_dial, emb]
    context_emb_cap = tf.concat([context_emb_cap, aggregated_lm_emb_cap], 2) # [num_candidates_cap, max_candidate_length_cap, emb]

    context_emb_dial = tf.nn.dropout(context_emb_dial, self.lexical_dropout) # [num_sentences_dial, max_sentence_length_dial, emb]
    context_emb_cap = tf.nn.dropout(context_emb_cap, self.lexical_dropout) # [num_candidates_cap, max_candidate_length_cap, emb]
    head_emb_cap = tf.nn.dropout(head_emb_cap, self.lexical_dropout) # [num_candidates_cap, max_candidate_length_cap, emb]

    context_emb_cdd = tf.concat([context_emb_cdd, aggregated_lm_emb_cdd], 2) # [num_candidates_cdd, max_candidate_length_cdd, emb]
    context_emb_cdd = tf.nn.dropout(context_emb_cdd, self.lexical_dropout) # [num_candidates_cdd, max_candidate_length_cdd, emb]
    head_emb_cdd = tf.nn.dropout(head_emb_cdd, self.lexical_dropout) # [num_candidates_cdd, max_candidate_length_cdd, emb]

    # len mask for caption and dialog
    text_len_dial = text_len[1:]
    text_len_mask_dial = tf.sequence_mask(text_len_dial, maxlen=max_sentence_length_dial) # [num_sentence_dial, max_sentence_length_dial]

    # extract lstm feature for cap and dial, and flatten to only valid words for dial
    context_outputs_cap = self.lstm_contextualize(context_emb_cap, text_len_cap) # [num_candidates_cap, max_candidate_length_cap, emb]
    context_outputs_dial = self.lstm_contextualize(context_emb_dial, text_len_dial, text_len_mask_dial) # [num_words_dial, emb]
    num_words_dial = util.shape(context_outputs_dial, 0)
    num_words = tf.reduce_sum(text_len)
    context_outputs = tf.concat([tf.zeros([num_words - num_words_dial, util.shape(context_outputs_dial, 1)]), context_outputs_dial], 0) # [num_words, emb]
    context_outputs_cdd = self.lstm_contextualize(context_emb_cdd, text_len_cdd) # [num_candidates_cdd, max_candidate_length_cdd, emb]

    # flatten head embedding of only valid words
    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    # keep candidates in dialog, exclude those in caption
    candidate_mask_dial = tf.logical_and(candidate_mask, candidate_starts >= text_len[0]) # [num_words, max_span_width]
    flattened_candidate_mask_dial = tf.reshape(candidate_mask_dial, [-1]) # [num_words * max_span_width]

    candidate_starts_dial = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask_dial) # [num_candidates_dial]
    candidate_ends_dial = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask_dial) # [num_candidates_dial]

    candidate_span_emb_dial = self.get_span_emb_dial(flattened_head_emb, context_outputs, candidate_starts_dial, candidate_ends_dial) # [num_candidates, emb]

    # get span emb of candidates in caption
    candidate_span_emb_cap = self.get_span_emb_phrases(head_emb_cap, context_outputs_cap, candidate_starts_caption, candidate_ends_caption) # [num_candidates, emb]

    candidate_ends_cdd = tf.cumsum(text_len_cdd) + num_words - 1
    candidate_starts_cdd = candidate_ends_cdd - text_len_cdd + 1
    candidate_span_emb_cdd = self.get_span_emb_phrases(head_emb_cdd, context_outputs_cdd, candidate_starts_cdd, candidate_ends_cdd) # [num_candidates, emb]

    if self.use_im:
      num_sentences_obj = tf.shape(context_word_emb_obj)[0]
      max_sentence_length_obj = tf.shape(context_word_emb_obj)[1]

      context_emb_list_obj = [context_word_emb_obj]
      head_emb_list_obj = [head_word_emb_obj]

      # get char embedding by conv1d on char embeddings of each word
      if self.config["char_embedding_size"] > 0:
        char_emb_obj = tf.gather(char_emb_all, char_index_obj) # [num_sentences, max_sentence_length, max_word_length, emb]
        flattened_char_emb_obj = tf.reshape(char_emb_obj, [num_sentences_obj * max_sentence_length_obj, util.shape(char_emb_obj, 2), util.shape(char_emb_obj, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
        flattened_aggregated_char_emb_obj = util.cnn(flattened_char_emb_obj, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
        aggregated_char_emb_obj = tf.reshape(flattened_aggregated_char_emb_obj, [num_sentences_obj, max_sentence_length_obj, util.shape(flattened_aggregated_char_emb_obj, 1)]) # [num_sentences, max_sentence_length, emb]
        context_emb_list_obj.append(aggregated_char_emb_obj)
        head_emb_list_obj.append(aggregated_char_emb_obj)

      context_emb_obj = tf.concat(context_emb_list_obj, 2) # [num_sentences, max_sentence_length, emb]
      head_emb_obj = tf.concat(head_emb_list_obj, 2) # [num_sentences, max_sentence_length, emb]

      num_candidates_obj = util.shape(lm_emb_obj, 0)
      max_candidate_length_obj = util.shape(lm_emb_obj, 1)
      flattened_lm_emb_obj = tf.reshape(lm_emb_obj, [num_candidates_obj * max_candidate_length_obj * lm_emb_size, lm_num_layers])
      flattened_aggregated_lm_emb_obj = tf.matmul(flattened_lm_emb_obj, tf.expand_dims(self.lm_weights, 1)) # [num_candidates_obj * max_candidate_length_obj * emb, 1]
      aggregated_lm_emb_obj = tf.reshape(flattened_aggregated_lm_emb_obj, [num_candidates_obj, max_candidate_length_obj, lm_emb_size])
      aggregated_lm_emb_obj *= self.lm_scaling

      context_emb_obj = tf.concat([context_emb_obj, aggregated_lm_emb_obj], 2) # [num_candidates_obj, max_candidate_length_obj, emb]
      context_emb_obj = tf.nn.dropout(context_emb_obj, self.lexical_dropout) # [num_candidates_obj, max_candidate_length_obj, emb]
      head_emb_obj = tf.nn.dropout(head_emb_obj, self.lexical_dropout) # [num_candidates_obj, max_candidate_length_obj, emb]

      context_outputs_obj = self.lstm_contextualize(context_emb_obj, text_len_obj) # [num_candidates_obj, max_candidate_length_obj, emb]

      candidate_ends_obj = tf.cumsum(text_len_obj) - 1
      candidate_starts_obj = candidate_ends_obj - text_len_obj + 1
      obj_span_emb = self.get_span_emb_phrases(head_emb_obj, context_outputs_obj, candidate_starts_obj, candidate_ends_obj) # [num_candidates, emb]

    # concat candidates in caption here
    candidate_starts = tf.concat([candidate_starts_cdd, candidate_starts_caption, candidate_starts_dial], 0)
    candidate_ends = tf.concat([candidate_ends_cdd, candidate_ends_caption, candidate_ends_dial], 0)
    candidate_span_emb = tf.concat([candidate_span_emb_cdd, candidate_span_emb_cap, candidate_span_emb_dial], 0) # [num_candidates, emb]
    candidate_cluster_ids_cap = self.get_candidate_labels(candidate_starts_caption, candidate_ends_caption, gold_starts, gold_ends, cluster_ids)
    candidate_cluster_ids_dial = self.get_candidate_labels(candidate_starts_dial, candidate_ends_dial, gold_starts, gold_ends, cluster_ids)
    candidate_cluster_ids = tf.concat([tf.zeros([util.shape(candidate_starts_cdd, 0)], tf.int32), candidate_cluster_ids_cap, candidate_cluster_ids_dial], 0) # [num_candidates]
    candidate_pool_flag = tf.cast(tf.concat([tf.ones(util.shape(candidate_starts_cdd, 0) + util.shape(candidate_starts_caption, 0), tf.int32), tf.zeros(util.shape(candidate_starts_dial, 0), tf.int32)], 0), tf.bool)

    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb) # [k, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    k = tf.minimum(tf.to_int32(tf.floor(tf.to_float(util.shape(candidate_starts, 0)) * self.config["top_span_ratio"])), tf.shape(candidate_mention_scores)[0])
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               util.shape(candidate_mention_scores, 0),
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]
    # coref_ops add extra 0 to top_span_indices, have to remove it here
    first_index = tf.gather(top_span_indices, tf.constant([0]))
    valid_indices = tf.boolean_mask(top_span_indices, tf.logical_not(tf.equal(top_span_indices, first_index)))
    top_span_indices = tf.concat([first_index, valid_indices], 0)
    k = util.shape(top_span_indices, 0)

    # rearrange top_span to put cdd and cap first
    top_span_cdd_pool_flag = tf.gather(candidate_pool_flag, top_span_indices) # [k]
    top_span_indices_cdd_cap = tf.boolean_mask(top_span_indices, top_span_cdd_pool_flag)
    top_span_indices_dial = tf.boolean_mask(top_span_indices, tf.logical_not(top_span_cdd_pool_flag))
    top_span_indices = tf.concat([top_span_indices_cdd_cap, top_span_indices_dial], 0)

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]

    top_span_cdd_pool_flag = tf.gather(candidate_pool_flag, top_span_indices) # [k]

    c = tf.minimum(self.config["max_top_antecedents"], k)

    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c, top_span_cdd_pool_flag)

    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    for i in range(self.config["coref_depth"]):
      if self.use_im:
        att_grid = self.get_span_im_emb(top_span_emb, obj_span_emb) # [k, emb], [k, emb]
      with tf.variable_scope("coref_layer", reuse=(i > 0)):
        top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
        if self.use_im:
          top_antecedent_scores_text, top_antecedent_scores_im = self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, att_grid, has_obj) # [k, c]
          top_antecedent_scores = top_fast_antecedent_scores + (1 - self.vis_weight) * top_antecedent_scores_text + self.vis_weight * top_antecedent_scores_im
        else:
          top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids) # [k, c]
        top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
        top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
        attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
        with tf.variable_scope("f"):
          f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
          top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]

    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
    same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]

    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    outputs = [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores,
     tokens_cdd, tokens_obj]
    if self.use_im:
      outputs.append(att_grid)
    else:
      outputs.append(tf.zeros([1, 1]))

    return outputs, loss

  def get_span_emb_dial(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      with tf.variable_scope("use_feature", reuse=tf.AUTO_REUSE):
        span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores", reuse=tf.AUTO_REUSE):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]

  def get_span_emb_phrases(self, head_emb, context_outputs, span_starts, span_ends):
    # context_outputs: [num_candidates_cap, max_candidate_length_cap, emb]
    # head_emb [num_candidates_cap, max_span_width_cap, emb]
    span_emb_list = []
    num_candidates = util.shape(context_outputs, 0)

    span_width = 1 + span_ends - span_starts # [num_candidates_cap]
    max_span_width = util.shape(context_outputs, 1)
    context_emb_size = util.shape(context_outputs, 2)

    context_outputs = tf.reshape(context_outputs, [-1, context_emb_size]) # [num_candidates_cap * max_candidate_length_cap, emb]
    span_start_indices = tf.range(num_candidates) * max_span_width # [num_candidates_cap]
    span_start_emb = tf.gather(context_outputs, span_start_indices) # [num_candidates_cap, emb]
    span_emb_list.append(span_start_emb)

    span_end_indices = span_start_indices + span_width - 1 # [num_candidates_cap]
    span_end_emb = tf.gather(context_outputs, span_end_indices) # [num_candidates_cap, emb]
    span_emb_list.append(span_end_emb)

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      with tf.variable_scope("use_feature", reuse=tf.AUTO_REUSE):
        span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      with tf.variable_scope("head_scores", reuse=tf.AUTO_REUSE):
        span_head_scores = util.projection(context_outputs, 1)  # [num_candidates_cap * max_span_width, 1]
      span_head_scores = tf.reshape(span_head_scores, [num_candidates, max_span_width, 1])
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, max_span_width, dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * head_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]

  def get_span_im_emb(self, span_emb, obj_span_emb):
    k = util.shape(span_emb, 0)
    n = util.shape(obj_span_emb, 0)
    with tf.variable_scope("image_attention", reuse=tf.AUTO_REUSE):
      # span_emb: [k, emb]
      map_dim = self.im_emb_size
      with tf.variable_scope("att_projection0"):
        text_map = util.projection(span_emb, map_dim) # [k, map_dim]
        obj_map = util.projection(obj_span_emb, map_dim) # [k, map_dim]
      text_map = tf.nn.relu(text_map)
      obj_map = tf.nn.relu(obj_map)

      text_map = tf.tile(tf.expand_dims(text_map, 1), [1, n, 1]) # [k, n, map_dim]
      obj_map = tf.tile(tf.expand_dims(obj_map, 0), [k, 1, 1]) # [k, n, map_dim]

      # interact via element wise map
      text_obj_combine = tf.nn.l2_normalize(text_map * obj_map, 2) # [k, n, map_dim]
      with tf.variable_scope("get_attention"):
        w_att = tf.get_variable('w_att', [map_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
      att_grid = tf.reshape(tf.matmul(tf.reshape(text_obj_combine, [-1, map_dim]), w_att), [k, n]) # [k, n]

      # softmax
      att_grid_soft = tf.nn.softmax(att_grid) # [k, n]

    return att_grid_soft # [k, n]

  def get_mention_scores(self, span_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, att_grid=None, has_obj=None):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb=1270]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb=3850]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]

    if self.use_im:
      # att max
      def zero_att_max(k):
        return tf.zeros([k, 1])

      def obj_att_max(att_grid):
        return tf.reduce_max(att_grid[:, :-1], axis=1, keepdims=True) #  [k, 1]

      top_span_att_max = tf.cond(has_obj, lambda: obj_att_max(att_grid), lambda: zero_att_max(k)) # [k, 1]
      top_antecedent_att_max = tf.gather(top_span_att_max, top_antecedents) # [k, c, 1]
      target_att_max = tf.expand_dims(top_span_att_max, 2) # [k, 1, 1]
      similarity_emb_att = top_antecedent_att_max * target_att_max # [k, c, 1]
      target_emb_att = tf.tile(target_att_max, [1, c, 1]) # [k, c, 1]

      # att similarity
      top_antecedent_att = tf.gather(att_grid, top_antecedents) # [k, c, n]
      top_span_att = tf.expand_dims(att_grid, 1) # [k, 1, n]

      def zero_ant_att_max(k, c):
        return tf.zeros([k, c, 1])

      def obj_ant_att_max(att_grid):
        return tf.reduce_max(att_grid[:, :, :-1], axis=2, keepdims=True) # [k, c, 1]

      top_span_antecedent_att_max = tf.cond(has_obj, lambda: obj_ant_att_max(top_antecedent_att * top_span_att), lambda: zero_ant_att_max(k, c))  # [k, c, 1]

      similarity_emb_att = tf.concat([similarity_emb_att, top_span_antecedent_att_max], 2) # [k, c, 2]
      
      pair_emb_im = tf.concat([target_emb_att, top_antecedent_att_max, similarity_emb_att], 2) # [k, c, 4 (+3n)]

      with tf.variable_scope("slow_antecedent_scores_im"):
        slow_antecedent_scores_im = util.ffnn(pair_emb_im, self.config["ffnn_depth_im"], self.config["ffnn_size_im"], 1, self.dropout) # [k, c, 1]
      slow_antecedent_scores_im = tf.squeeze(slow_antecedent_scores_im, 2) # [k, c]

      return slow_antecedent_scores, slow_antecedent_scores_im # [k, c]
    
    return slow_antecedent_scores # [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask=None):
    num_sentences = tf.shape(text_emb)[0]
    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fw_cell", reuse=tf.AUTO_REUSE):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell", reuse=tf.AUTO_REUSE):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs # [num_sentences, max_sentence_length, emb]

    if text_len_mask is None:
      return text_outputs
    else:
      return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def get_predicted_clusters_attention(self, top_span_starts, top_span_ends, att_grid, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))

      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    # att_grid is the same order as top_span, extract them for each mention in predicted_clusters
    predicted_att_grids = []
    for cluster in predicted_clusters:
      att_grid_cluster = []
      for mention in cluster:
        find_mention = False
        for index, (start, end) in enumerate(zip(top_span_starts, top_span_ends)):
          if mention[0] == start and mention[1] == end:
            att_grid_cluster.append(att_grid[index])
            find_mention = True
            break
        if not find_mention:
          raise ValueError('antecedent not found in top spans')
      predicted_att_grids.append(att_grid_cluster)

    return predicted_clusters, predicted_att_grids, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print(f"Loaded {len(self.eval_data)} eval examples.")

  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    coref_predictions = {}
    pr_coref_evaluator = metrics.PrCorefEvaluator()

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}

      outputs = session.run(self.predictions, feed_dict=feed_dict)
      candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, tokens_cdd, tokens_obj, att_grid = outputs

      predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"])
      pr_coref_evaluator.update(coref_predictions[example["doc_key"]], example["pronoun_info"], example["sentences"], tokens_cdd)
      if example_num % 50 == 0:
        print(f"Evaluated {example_num + 1}/{len(self.eval_data)} examples.")

    summary_dict = {}
    pr_coref_results = pr_coref_evaluator.get_prf()

    summary_dict["Pronoun Coref average F1 (py)"] = pr_coref_results['f']
    print(f"Pronoun Coref average F1 (py): {pr_coref_results['f'] * 100:.2f}%")
    summary_dict["Pronoun Coref average precision (py)"] = pr_coref_results['p']
    print(f"Pronoun Coref average precision (py): {pr_coref_results['p'] * 100:.2f}%")
    summary_dict["Pronoun Coref average recall (py)"] = pr_coref_results['r']
    print(f"Pronoun Coref average recall (py): {pr_coref_results['r'] * 100:.2f}%")

    summary_dict["Discussed Pronoun Coref average F1 (py)"] = pr_coref_results['f_discussed']
    print(f"Discussed Pronoun Coref average F1 (py): {pr_coref_results['f_discussed'] * 100:.2f}%")
    summary_dict["Discussed Pronoun Coref average precision (py)"] = pr_coref_results['p_discussed']
    print(f"Discussed Pronoun Coref average precision (py): {pr_coref_results['p_discussed'] * 100:.2f}%")
    summary_dict["Discussed Pronoun Coref average recall (py)"] = pr_coref_results['r_discussed']
    print(f"Discussed Pronoun Coref average recall (py): {pr_coref_results['r_discussed'] * 100:.2f}%")

    summary_dict["Not Discussed Pronoun Coref average F1 (py)"] = pr_coref_results['f_not_discussed']
    print(f"Not Discussed Pronoun Coref average F1 (py): {pr_coref_results['f_not_discussed'] * 100:.2f}%")
    summary_dict["Not Discussed Pronoun Coref average precision (py)"] = pr_coref_results['p_not_discussed']
    print(f"Not Discussed Pronoun Coref average precision (py): {pr_coref_results['p_not_discussed'] * 100:.2f}%")
    summary_dict["Not Discussed Pronoun Coref average recall (py)"] = pr_coref_results['r_not_discussed']
    print(f"Not Discussed Pronoun Coref average recall (py): {pr_coref_results['r_not_discussed'] * 100:.2f}%")

    average_f1 = pr_coref_results['f']
    max_eval_f1 = tf.maximum(self.max_eval_f1, average_f1)
    self.update_max_f1 = tf.assign(self.max_eval_f1, max_eval_f1)

    return util.make_summary(summary_dict), average_f1
