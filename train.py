#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

import os
import time
import subprocess
import sys

import tensorflow as tf
from model import VisCoref
import util

def set_log_file(fname):
  tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
  os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
  os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
  config = util.initialize_from_env()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  log_file = os.path.join(log_dir, 'train.log')
  set_log_file(log_file)

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  
  tf.set_random_seed(config['random_seed'])

  model = VisCoref(config)
  saver = tf.train.Saver()

  max_f1 = 0

  config_tf = tf.ConfigProto()
  config_tf.gpu_options.allow_growth = True
  with tf.Session(config=config_tf) as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print(f"Restoring from: {ckpt.model_checkpoint_path}")
      saver.restore(session, ckpt.model_checkpoint_path)
      max_f1 = session.run(model.max_eval_f1)
      print(f'Restoring from max f1 of {max_f1:.2f}')

    initial_time = time.time()

    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step == 1 or tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print(f"[{tf_global_step}] loss={average_loss:.4f}, steps/s={steps_per_second:.2f}")
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step == 1 or tf_global_step % eval_frequency == 0:
        eval_summary, eval_f1 = model.evaluate(session)
        _ = session.run(model.update_max_f1)
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        
        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
        
        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        print(f"[{tf_global_step}] evaL_f1={eval_f1:.2f}, max_f1={max_f1:.2f}")

        if tf_global_step >= config['max_step']:
          print('Training finishes due to reaching max steps')
          break

