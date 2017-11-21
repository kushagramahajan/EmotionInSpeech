from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn as tflearn
import wave

from pathlib import Path
slim = tf.contrib.slim

def get_annotation():
  label_path='/home/kushagra/RECOLA-Annotation/emotional_behaviour/arousal/*'
  label_path = Path(label_path)
  input_arr=[]
  print('Extracting labels from {}'.format(label_path))
  
  paths=[]
  for path in label_path.parent.glob(label_path.name):
      paths.append(path)

  paths.sort()
  print(paths)

  for path in paths:
      print(path)
      #print(path.suffixes[-2])
      portion = path.suffixes[-1][1:]
      print('Processing {}'.format(path))
      gts=[]
      counter=0
      with open(str(path)) as f:
          #gts = [np.array(l.strip().split(',')) for l in f.readlines() if l[0] != '@']
          for l in f.readlines():
              counter+=1
              if(counter<=2):
                  continue
              l.replace('\r\n','')
              #l=l.rstrip()
              if(len(l)>2):
                  #print(l.split(','))
                  gts.append([l.split(';')[1:]])
#      gts=np.array(gts)
      input_arr.append(gts)
#      print(input_arr)
      #print(len(input_arr[0][0][0]))
      #break

  input_arr=np.array(input_arr)
  input_arr=np.swapaxes(input_arr,2,3)
  input_arr=np.squeeze(input_arr, axis=3)
  print(input_arr.shape)
  return input_arr

def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

def recurrent_model(net, hidden_units=64, num_lstm_modules=2):
  """Adds the LSTM network on top of the spatial audio model.

  Args:
     net: A `Tensor` of dimensions [batch_size, seq_length, num_features].
     hidden_units: The number of hidden units of the LSTM cell.
     num_classes: The number of classes.
  Returns:
      The prediction of the network.
  """

  batch_size, seq_length, num_features = net.get_shape().as_list()

  lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                                 use_peepholes=True,
                                 cell_clip=100,
                                 state_is_tuple=True)

  stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [lstm] * num_lstm_modules, state_is_tuple=True)


  outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)
  ##addded by kush 
  #y=get_annotation()
  print('output shape before dnn layer: ',outputs.get_shape())
  #outputs = dnn_layers(outputs[-1], None)
  print('outputs[-1]: ',outputs[-1].get_shape())
  print('outputs shape: ',outputs.get_shape())
  print('shape of y: ',y.get_shape())
  #outputs=tf.reshape(outputs, (batch_size*13271, hidden_units))
  #prediction, loss = tflearn.models.linear_regression(outputs, y)
  # We have to specify the dimensionality of the Tensor so we can allocate
  # weights for the fully connected layers.
  ###modified by kush
  net = tf.reshape(outputs[:,-1], (batch_size, hidden_units))
  print('net shape: ',net.get_shape()) 
  ## commented by kush
  prediction = slim.layers.linear(net, 6)
  print('gvhgvhgvhvhv prediction shape: ',prediction.get_shape())
  
  ##modified by kush
  return tf.reshape(prediction, (batch_size,6))
  #return prediction


def audio_model(inputs, conv_filters=32, num_layers=8):
    """Creates the audio model.

    Args:
        inputs: A tensor that contains the audio input.
        conv_filters: The number of convolutional filters to use.
    Returns:
        The audio model.
    """
    ##changed by kush
    batch_size,_, num_features = inputs.get_shape().as_list()
    seq_length = tf.shape(inputs)[1]

    # return  tf.reshape(inputs, (batch_size, seq_length, num_features))

    net = tf.reshape(inputs, [batch_size * seq_length, 1, num_features, 1])
    net = tf.nn.avg_pool(
        net,
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 2, 1],
        padding='SAME',
        name='subsampling')
    
    with slim.arg_scope([slim.layers.conv2d],
                         padding='SAME', activation_fn=slim.batch_norm):
        for i in range(num_layers):
            net = slim.layers.conv2d(net, conv_filters, (1, 40))

            net = tf.nn.max_pool(
                net,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 2, 1],
                padding='SAME',
                name='pool')
           
    num_features = np.multiply(*net.get_shape().as_list()[-2:])
    net = tf.reshape(net, (batch_size, seq_length, num_features))
    return net



def get_model(name,ground_truth):
  """ Returns the recurrent audio model.

  Args:
      name: The model to return. Here only 'audio'.
  Returns:
      The recurrent audio model.
  """
  global y
  y=ground_truth
  print('shape of y in get_model: ',y.get_shape())
  name_to_fun = {'audio': audio_model}

  if name in name_to_fun:
    model = name_to_fun[name]
  else:
    raise ValueError('Requested name [{}] not a valid model'.format(name))

  def wrapper(*args, **kwargs):
    return recurrent_model(model(*args), **kwargs)

  return wrapper

