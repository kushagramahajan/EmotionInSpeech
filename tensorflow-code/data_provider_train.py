from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pathlib import Path

slim = tf.contrib.slim


class Dataset:
    #num_classes = 2

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
    def get_split(self, split_name='train', batch_size=5):
      """Returns a data split of the ComParE dataset.

      Args:
          dataset_dir: The directory that contains the data.
          split_name: One or more train/test/valid split names.
          batch_size: The size of the batch.
      Returns:
          The raw audio examples and the corresponding arousal/valence
          labels.
      """

      paths = [str(Path(self.dataset_dir) / '{}audioseq150train19.tfrecords'.format(name)) 
               for name in split_name.split(',')]
      print('paths in provider_kush: ',paths)
      is_training = 'train' in split_name

      filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

      reader = tf.TFRecordReader()

      _, serialized_example = reader.read(filename_queue)

      features = tf.parse_single_example(
              serialized_example,
              features={
                  'audio_raw': tf.FixedLenFeature([], tf.string),
                  'annotation_raw': tf.FixedLenFeature([], tf.string),
                  }
              )

      raw_audio = features['audio_raw']
      label = features['annotation_raw']
#      print('raw audio in provider: ',raw_audio[0])
#      if is_training:
#        raw_audio, label = tf.train.shuffle_batch(
#                [raw_audio, label], 1, 1000, 100, 4)
#        raw_audio = raw_audio[0]
#        label = label[0]

#      print(label) 
      
      raw_audio = tf.decode_raw(raw_audio, tf.float32)
      label = tf.decode_raw(label, tf.float32)
      print('raw_audio shape: ',raw_audio.shape)
      print('label shape: ',label.shape)
      print(raw_audio)
      print(label) 
#      if is_training:
#        raw_audio += tf.random_normal(tf.shape(raw_audio), stddev=.25)

      raw_audio=tf.reshape(raw_audio,(150,640))
      label=tf.reshape(label,(150,6))
      frames, labels = tf.train.batch([raw_audio, label], batch_size,
                                    capacity=1000, dynamic_pad=True)

      print('frames shape: ',frames.get_shape().as_list())
      frames = tf.reshape(frames, (batch_size, -1, 640))
      print('frames shape afte reshape: ',frames.get_shape().as_list())
      print('labels shape: ',labels.get_shape().as_list())
     # labels = tf.reshape(labels, (batch_size,6))
      print('labels shape after reshape: ',labels.get_shape().as_list())
      
      ### commented by kushagra
      #labels = slim.one_hot_encoding(labels, self.num_classes)

#      seq_length=7501
#      ground_truth_actual = [[0 for x in range(6)] for y in range(batch_size)] 
#      sess=tf.Session()
      #with sess.as_default():
#      tf.InteractiveSession()
#      for i in range(batch_size):
#          for j in range(6):
#              ground_truth_actual[i][j]=labels[i][seq_length-1][j].eval()
#      ground_truth_actual=tf.constant(ground_truth_actual)
#      ground_truth_actual=np.array(ground_truth_actual)
#      print('groound_truth_actual shape: ',ground_truth_actual.get_shape())
      




      return frames, labels, sum(self._split_to_num_samples[name] for name in split_name.split(','))

    
class AddresseeProvider(Dataset):
    num_classes = 2
    _split_to_num_samples = {
      'test': 3594,
      'devel': 3550,
      'train': 3742
    }

    
class ColdProvider(Dataset):
    num_classes = 2
    _split_to_num_samples = {
      'test': 9551, 
      'devel': 9596, 
      'train': 9505
    }


class SnoreProvider(Dataset):
    num_classes = 4
    _split_to_num_samples = {
      'test': 500, 
      'devel': 644, 
      'train': 500
    }

class ArousalProvider(Dataset):
    _split_to_num_samples = {
      'test': 500,
      'devel': 200,
      'train': 5000
    }



def get_provider(name):
  """Returns the provider with the given name

  Args:
      name: The provider to return. Here only 'cacac' or 'urtic'.
  Returns:
      The requested provider.
  """

  name_to_class = {'addressee': AddresseeProvider, 'cold': ColdProvider,
                   'snore': SnoreProvider, 'arousal': ArousalProvider}

  if name in name_to_class:
    provider = name_to_class[name]
  else:
    raise ValueError('Requested name [{}] not a valid provider'.format(name))

  return provider

