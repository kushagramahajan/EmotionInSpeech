from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_provider_train
import models_train
import data_generator_wav
from tensorflow.python.platform import tf_logging as logging
import numpy as np

slim = tf.contrib.slim

def batch_generator(data, batch_size, shuffle=False):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]




FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 50, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'How many preprocess threads to use.')
tf.app.flags.DEFINE_string('train_dir', 'ckpt2/',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 50000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('num_lstm_modules', 2, 'Number of LSTM modules to use.')
tf.app.flags.DEFINE_string('train_device', '/gpu:0', 'Device to train with.')
tf.app.flags.DEFINE_string('model', 'audio',
                           '''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/home/kushagra/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('task', 'arousal', 'The task to execute. `addressee`, `cold`, or `snore`')
tf.app.flags.DEFINE_string('portion', 'train', 'Portion to use for training.')


def train():
  """Trains the audio model.

  Args:
     data_folder: The folder that contains the trainin data.
  """
  
  
  g = tf.Graph()
   
  seq_length=13

  with g.as_default():
    # Load dataset.
    provider = data_provider_train.get_provider(FLAGS.task)
    audio, ground_truth, _ = provider(
        FLAGS.dataset_dir).get_split(FLAGS.portion)
    #print(ground_truth.initialized_value())
    print('back from providerr')
    print('ground_truth shape: ',ground_truth)
    #ground_truth_actual = [[0 for x in range(6)] for y in range(FLAGS.batch_size)] 
    #sess=tf.Session(graph=g)
    #with sess.as_default():
    #  for i in range(FLAGS.batch_size):
    #      for j in range(6):
    #         ground_truth_actual[i][j]=ground_truth[i][seq_length-1][j].eval()
    #  ground_truth_actual=tf.constant(ground_truth_actual)
    #print('groound_truth_actual shape: ',ground_truth_actual.get_shape())
    #sess.close()
#    ann=data_generator_wav.get_annotation()
#    ann = ann.astype(np.float)
#    audio_all=data_generator_wav.serialize_sample_return()
#    ann=np.reshape(ann,(172523,6)) 
#    audio_all=np.reshape(audio_all,(172523,640))
#    audio, ground_truth = tf.train.batch([audio_all, ann], batch_size=13,
#                                     dynamic_pad=True)
     ####added by kush
    
#    audio=np.reshape(audio_all,(FLAGS.batch_size,-1,640))
#    ground_truth=np.reshape(annotation,(FLAGS.batch_size,-1,6))
    
    #audio = tf.placeholder(tf.float32, [FLAGS.batch_size,seq_length ,640])
    #ground_truth_actual = tf.placeholder(tf.float32, [FLAGS.batch_size ,6])
    print('ground truth in compare train shape: ',ground_truth.get_shape())
    print('audio in compare train shape: ',audio.get_shape())
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                        is_training=True):
        prediction = models_train.get_model(FLAGS.model,ground_truth)(audio,num_lstm_modules=FLAGS.num_lstm_modules)

    print('after prediction in compare_train_kush')


    ###modified by kush
#    ground_truth=tf.reshape(ground_truth, (FLAGS.batch_size*13271,6))
    
    
    #loss = tf.nn.weighted_cross_entropy_with_logits(prediction, ground_truth
    #                                                            ,pos_weight=1)
    #ground_truth_actual = [[0 for x in range(6)] for y in range(FLAGS.batch_size)] 
    #for i in range(FLAGS.batch_size):
    #    for j in range(6):
    #	    ground_truth_actual[i][j]=ground_truth[i][seq_length-1][j].eval()
    #ground_truth_actual=np.array(ground_truth_actual)
    #print('ground_truth_actual: ',ground_truth_actual)
    #ground_truth_actual=tf.constant(ground_truth_actual)
    
    loss=slim.losses.mean_squared_error(prediction, ground_truth[:,seq_length-1,:])
    #loss = slim.losses.compute_weighted_loss(loss)
    #loss=tf.metrics.mean_squared_error(ground_truth_actual,prediction)
    total_loss = slim.losses.get_total_loss()
    #total_loss = tf.losses.get_total_loss()
    
    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(ground_truth[:,seq_length-1,:], 1), tf.argmax(prediction, 1))))
    
    chance_accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(ground_truth[:,seq_length-1,:], 1), 0)))
    
    tf.summary.scalar('losses/total loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('chance accuracy', chance_accuracy)
    #tf.histogram_summary('labels', tf.argmax(ground_truth, 1))
    tf.summary.scalar('losses/Cross Entropy Loss', loss)

    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

#    with tf.Session(graph=g) as sess:
#        print('inside session')
#        if FLAGS.pretrained_model_checkpoint_path:
#            variables_to_restore = slim.get_variables_to_restore()
#            saver = tf.train.Saver(variables_to_restore)
#            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

#        train_op = slim.learning.create_train_op(total_loss,
#                                                 optimizer,
#                                                 summarize_gradients=True)
        
#        logging.set_verbosity(1)
#        slim.learning.train(train_op,
#                            FLAGS.train_dir,
#                            save_summaries_secs=60,
#                            save_interval_secs=600)


  
  with tf.Session(graph=g) as sess1:
      #ann=data_generator_wav.get_annotation()
      #ann = ann.astype(np.float)
      #audio_all=data_generator_wav.serialize_sample_return()
      #ann=np.reshape(ann,(172523,6))
      #audio_all=np.reshape(audio_all,(172523,640))
      #print('annotation in compare train shape: ',ann.shape)
      #print('audio_all in compare train shape: ',audio_all.shape)

      tf.initialize_all_variables().run()    
      #gen_source_only_batch = batch_generator(
      #      [audio_all, ann], FLAGS.batch_size*seq_length)


      logging.set_verbosity(1)
     # for i in range(5):
      #audio1,ann=gen_source_only_batch.next()
      #ground_truth = np.reshape(ann, (FLAGS.batch_size,seq_length ,6))
      #audio1 = np.reshape(audio1, (FLAGS.batch_size,seq_length ,640))
      #ground_truth_actual1 = [[0 for x in range(6)] for y in range(FLAGS.batch_size)] 
      #for i in range(FLAGS.batch_size):
      #    for j in range(6):
      #        ground_truth_actual1[i][j]=ground_truth[i][seq_length-1][j]
      #ground_truth_actual1=np.array(ground_truth_actual1)
      #print('audio shape in loop: ',audio1.shape)
      #print('ground truth shape: ',ground_truth.shape)
      #print('ground_truth_actual shape in loop: ',ground_truth_actual1.shape)
	  #audio=tf.constant(audio)
          #ground_truth_actual=tf.constant(ground_truth_actual)
         # l,tl,acc=sess1.run([loss,total_loss,accuracy],feed_dict={audio:audio1,ground_truth_actual:ground_truth_actual1})	
      print('inside session')

      train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)
      if FLAGS.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess1, FLAGS.pretrained_model_checkpoint_path)


      slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)
def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
