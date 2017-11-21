from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from pathlib import Path
import wave
from keras.preprocessing import sequence
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, AveragePooling2D, AveragePooling1D, Reshape, Flatten, Conv1D
import tensorflow as tf
#from keras.datasets import cifar10
#from keras.datasets import imdb
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
__signal_framerate = 16000
batch_size=1
epochs = 140
#maxlen = 100
conv_filters=40
num_layers=8
n_prev=1

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
      portion = path.suffixes[-1][1:]
      print('Processing {}'.format(path))
      gts=[]
      counter=0
      with open(str(path)) as f:
          for l in f.readlines():
              counter+=1
              if(counter<=1):
                  continue
              if(len(l)>2):
                  l=l[:-1]
                  gts.append([l.split(';')[1:]])
      input_arr.append(gts)

  input_arr=np.array(input_arr)
  input_arr=np.swapaxes(input_arr,2,3)
  input_arr=np.squeeze(input_arr, axis=3)
  print(input_arr.shape)
  return input_arr[:,1:,:]



def read_wave(path):
  """Reads a wav file and splits it in chunks of 40ms. 
  Pads with zeros if duration does not fit exactly the 40ms chunks.
  Assumptions: 
      A. Wave file has one channel.
      B. Frame rate of wav file is 16KHz.
  
  Args:
      wav_file: The name of the wav file.
  Returns:
      A data array, where each row corresponds to 40ms.
  """

  fp = wave.open(str(path))
  num_of_channels = fp.getnchannels()
  fps = fp.getframerate()

  if num_of_channels > 1:
    raise ValueError('The wav file should have 1 channel. [{}] found'.format(num_of_channels))

  if fps != __signal_framerate:
    raise ValueError('The wav file should have 16000 fps. [{}] found'.format(fps))

  chunk_size = 640 # 40ms if fps = 16k.

  num_frames = fp.getnframes()
  dstr = fp.readframes(num_frames * num_of_channels)
  data = np.fromstring(dstr, np.int16)
  audio = np.reshape(data, (-1))
  audio = audio / 2.**15 # Normalise audio data (int16).

  audio = np.pad(audio, (0, chunk_size - audio.shape[0] % chunk_size), 'constant')
  audio = audio.reshape(-1, chunk_size)

  return audio.astype(np.float32)


def serialize_sample():
  label_path='/home/kushagra/RECOLA-Audio-recordings-new/*'
  label_path = Path(label_path)
  audio_paths=[]
  all_audio=[]

  for path in label_path.parent.glob(label_path.name):
      audio_paths.append(path)

  audio_paths.sort()
  for i in range(len(audio_paths)):
    print(i)
    audio_path=audio_paths[i]
    audio_raw = read_wave(audio_path)
    print('raw_audio_shape: ',audio_raw.shape)
    all_audio.append(audio_raw)

  all_audio=np.array(all_audio)
  print('all_audio shape: ',all_audio.shape)
  return all_audio[:,1:,:]

def _load_data(dataX, dataY):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    i=0
    while 1:
        i+=1
#	if(i%50>(50-n_prev)):
#            continue
        if(i+n_prev>=len(dataX)):
	    i=0
        docX.append([dataX[i:i+n_prev]])
        docY.append([dataY[i+n_prev]])
        if(i%batch_size==0):
	    
            alsX = np.array(docX)
            alsY = np.array(docY)

            alsX=np.squeeze(alsX, axis=1)
            alsY=np.squeeze(alsY, axis=1)
            docX=[]
            docY=[]
	    alsY=np.reshape(alsY,(batch_size,150*6))
            #print(alsX.shape, alsY.shape)
            yield alsX,alsY
    #return alsX, alsY


annotation=get_annotation()
all_audio=serialize_sample()

all_audio=np.reshape(all_audio,(23*7500,640))
annotation=np.reshape(annotation, (23*7500,6))
annotation = annotation.astype(np.float)
all_audio = all_audio.astype(np.float)

for i in range(len(all_audio[0])):
  all_audio[:,i]=(all_audio[:,i]-np.mean(all_audio[:,i]))/float(np.std(all_audio[:,i]))

all_audio=np.reshape(all_audio,(23,7500,640))
annotation=np.reshape(annotation,(23,7500,6))


x_train=all_audio[0:19]
y_train=annotation[0:19]
x_test=all_audio[19:23]
y_test=annotation[19:23]
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_train=np.reshape(x_train,(19*7500,640))
x_train=np.reshape(x_train,(-1,96000))

x_test=np.reshape(x_test,(4*7500,640))
x_test=np.reshape(x_test,(-1,96000))

y_train=np.reshape(y_train,(19*7500,6))
y_train=np.reshape(y_train,(-1,150,6))

y_test=np.reshape(y_test,(4*7500,6))
y_test=np.reshape(y_test,(-1,150,6))

print('x_train shape: ', x_train.shape)
print(x_train.shape[1:])

def main_code(x_train, y_train, x_test, y_test):
  model = Sequential()
  model.add(Reshape((n_prev,96000,1),input_shape=(n_prev,96000)))
  print('model output after init reshape: ',model.outputs)
  
  model.add(Conv2D(conv_filters,(1,80), padding='same'))
  print('model output after conv1 layer: ',model.outputs)
  model.add(MaxPooling2D(pool_size=(1,2), strides=None))
  print('model output after max pooling 1 layer: ',model.outputs)


  model.add(Conv2D(conv_filters,(1,4000), padding='same'))
  print('model output after conv2 layer: ',model.outputs)

  model.add(Reshape((n_prev,48000,40)))
  model.add(MaxPooling2D(pool_size=(1,20), strides=None,data_format='channels_first'))
  print('model output after max pooling 2 layer: ',model.outputs)

  model.add(Reshape((150,-1)))
  model.add(LSTM(128,return_sequences=True))
  model.add(LSTM(128))
  print('model output after second lstm layer: ',model.outputs)
  model.add(Dense(150*6, activation='sigmoid'))
  print('model output after dense layer: ',model.outputs)
  model.compile(loss='mse', optimizer='rmsprop', lr=0.0001,decay=0.1)

  filepath='bsize1_fitgen_newmodel_2.h5'
  checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  callbacks_list = [checkpoint]

#  x_train1, y_train1 = _load_data(x_train, y_train)
#  x_test1, y_test1 = _load_data(x_test, y_test)
#  print('final shape x_train: ',x_train1.shape)
#  print('final shape y_train: ',y_train1.shape)

  
  my_generator=_load_data(x_train,y_train)
  my_generator_validation=_load_data(x_test, y_test)
  model.fit_generator(my_generator,
          steps_per_epoch=940,
          epochs=epochs,
          callbacks=callbacks_list, validation_data=my_generator_validation,validation_steps=190)


main_code(x_train, y_train, x_test, y_test)
