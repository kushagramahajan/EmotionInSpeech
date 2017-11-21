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

os.environ["CUDA_VISIBLE_DEVICES"]=""

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())



batch_size=150
__signal_framerate = 16000


def _load_data(dataX, dataY, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(dataX)-n_prev):
        if(i%7501>(7501-n_prev)):
	    continue
        docX.append([dataX[i:i+n_prev]])
        docY.append([dataY[i+n_prev]])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY



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
  return input_arr


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
  return all_audio

annotation=get_annotation()
all_audio=serialize_sample()

all_audio=np.reshape(all_audio,(23*7501,640))
annotation=np.reshape(annotation, (23*7501,6))
annotation = annotation.astype(np.float)
all_audio = all_audio.astype(np.float)

for i in range(len(all_audio[0])):
  all_audio[:,i]=(all_audio[:,i]-np.mean(all_audio[:,i]))/float(np.std(all_audio[:,i]))

all_audio=np.reshape(all_audio,(23,7501,640))
annotation=np.reshape(annotation,(23,7501,6))


x_train=all_audio[0:19]
y_train=annotation[0:19]
x_test=all_audio[19:23]
y_test=annotation[19:23]
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_train=np.reshape(x_train,(19*7501,640))
x_test=np.reshape(x_test,(4*7501,640))
y_train=np.reshape(y_train,(19*7501,6))
y_test=np.reshape(y_test,(4*7501,6))
print('x_train shape: ', x_train.shape)
print(x_train.shape[1:])


model = load_model('bsize100_fitgen.h5')
model.compile(loss='mse', optimizer='rmsprop')

x_test1, y_test1 = _load_data(x_test, y_test)
x_test1=np.squeeze(x_test1, axis=1)
y_test1=np.squeeze(y_test1, axis=1)
print('final shape x_test: ',x_test1.shape)
print('final shape y_test: ',y_test1.shape)

y_pred=model.predict(x_test1, batch_size=batch_size, verbose=0)

u= ((y_test1 - y_pred) ** 2).sum()/(len(y_test1)*len(y_test1[0]))

#avgloss=float(u)/len(x_test1)
#print('average loss: ',avgloss)

#v= ((y_test1 - y_test1.mean()) ** 2).sum()/(len(y_test1)*len(y_test1[0]))
#score=1-(u/v)

#print('u: ',u)
#print('v: ',v)
#print('score: ',score)

ux=y_test1.mean()
uy=y_pred.mean()
sigx=np.var(y_test1)
sigy=np.var(y_pred)
sigxy=np.cov(y_test1,y_pred)
lc=1-(2*sigxy/(sigx+sigy+(ux-uy)*(ux-uy)))
print('1:',ux, ux)
print('2:',sigx,sigy)
print('3: ',sigxy)
print('lc: ',lc.mean())
