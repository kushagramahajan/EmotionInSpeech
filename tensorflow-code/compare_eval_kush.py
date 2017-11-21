from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider_kush
import models_kush
import math

#from menpo.visualize import print_progress
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',5, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio, video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/home/kushagra/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckptseq150/', 'The checkpoint directory.')
tf.app.flags.DEFINE_string('log_dir', 'ckptseq150/eval/', 'The directory to save the event files.')
tf.app.flags.DEFINE_integer('num_lstm_modules', 2, 'Number of LSTM modules to use.')
tf.app.flags.DEFINE_integer('num_examples', None, 'The number of examples in the given portion.')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('portion', 'devel', 'The portion of the dataset to use -- `train`, `devel`, or `test`.')
tf.app.flags.DEFINE_string('task', 'arousal', 'The task to execute. `addressee`, `cold` or `snore`')

def evaluate(data_folder):
  """Evaluates the audio model.

  Args:
     data_folder: The folder that contains the data to evaluate the audio model.
  """
  seq_length=150
  g = tf.Graph()
  with g.as_default():
    # Load dataset.
    provider = data_provider_kush.get_provider(FLAGS.task)(data_folder)
    num_classes = 6
    audio, ground_truth, num_examples = provider.get_split(FLAGS.portion, FLAGS.batch_size)

    # Define model graph.
    with slim.arg_scope([slim.batch_norm],
                           is_training=False):
      predictions = models_kush.get_model(FLAGS.model,ground_truth)(audio, num_lstm_modules=FLAGS.num_lstm_modules)
      #pred_argmax = tf.argmax(predictions, 1) 
      #lab_argmax = tf.argmax(labels, 1)
	
      metrics = {
          "eval/accuracy": slim.metrics.streaming_mean_squared_error(predictions, ground_truth[:,seq_length-1,:])
      }

      total_error = tf.reduce_sum(tf.square(tf.subtract(ground_truth[:,seq_length-1,:], tf.reduce_mean(ground_truth[:,seq_length-1,:]))))
      unexplained_error = tf.reduce_sum(tf.square(tf.subtract(ground_truth[:,seq_length-1,:], predictions)))
      R_squared = tf.subtract(tf.cast(1, tf.float32), tf.divide(total_error, unexplained_error))
      print('R_squared value: ',R_squared)
      for i in range(num_classes):
          name ='eval/mse_{}'.format(i)
          recall = slim.metrics.streaming_mean_squared_error(predictions[:,i],ground_truth[:,seq_length-1,i])
          metrics[name] = recall

      metrics['R_squared']=(R_squared,tf.subtract(tf.cast(1, tf.float32), tf.div(total_error, unexplained_error)))
      #print(zip(metrics.values()))
      #metric_names = metrics.keys()
      #value_ops, update_ops = zip(*metrics.values())
      #names_to_values, names_to_updates = dict(zip(metric_names, value_ops)), dict(zip(metric_names, update_ops))
      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metrics)
      
      summary_ops = []
      metrics = dict()
      for name, value in names_to_values.items():
        op = tf.summary.scalar(name, value)
        op = tf.Print(op, [value], name)
        summary_ops.append(op)
        metrics[name] = value

      # Computing the unweighted average recall and add it into the summaries.
      uar = sum([metrics['eval/mse_{}'.format(i)] for i in range(num_classes)]) / num_classes
      op = tf.summary.scalar('eval/mse', uar)
      op = tf.Print(op, [uar], 'eval/mse')
      summary_ops.append(op)

      num_examples = FLAGS.num_examples or num_examples
      num_batches = math.ceil(num_examples / float(FLAGS.batch_size))
      logging.set_verbosity(1)

      # Setup the global step.
      slim.get_or_create_global_step()

      # How often to run the evaluation.
      eval_interval_secs = FLAGS.eval_interval_secs 

      slim.evaluation.evaluation_loop(
          '',
          FLAGS.checkpoint_dir,
          FLAGS.log_dir,
          num_evals=num_batches,
          eval_op=list(names_to_updates.values()),
          summary_op=tf.summary.merge(summary_ops),
          eval_interval_secs=eval_interval_secs)

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()




