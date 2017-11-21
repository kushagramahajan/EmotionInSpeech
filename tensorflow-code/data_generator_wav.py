import tensorflow as tf
import numpy as np
import wave

from pathlib import Path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('wave_folder', 'wav/', 'The folder that contains the wav files.')
tf.app.flags.DEFINE_string('arff_path', 'ComPaRe*arff', 'The glob for all the arff files of the datset.')
tf.app.flags.DEFINE_string('tf_folder', 'tf_records', 'The folder to write the tf records.')

__signal_framerate = 16000


def get_audios():
  label_path='/home/kushagra/RECOLA-Audio-recordings-new/*'
  label_path = Path(label_path)
  for i in label_path.parent.glob(label_path.name):
      print(i)
  for path in label_path.parent.glob(label_path.name):
      print(path)


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
              if(counter<=1):
                  continue
#	      l.replace('\r\n','')
	      #l=l.rstrip()
              if(len(l)>2):
                  l=l[:-1]
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

def get_labels(label_path):
  """Parses the data arff files to extract the labels 

  Args:
      label_path: A path glob which contains the arff files with the labels.
  Returns:
      A dictionary for the labels of each fold.
  """
  labels = {}
  class_names = None
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
      with open(str(path)) as f:
          #gts = [np.array(l.strip().split(',')) for l in f.readlines() if l[0] != '@']
          for l in f.readlines():
	      l.replace("\n","")
              l.replace("\r","")
	      if(l[0]!='@' and len(l)>2):
		  #print(l.split(','))
                  l=l[:-2]
	          gts.append([l.split(',')])
#      gts=np.array(gts)
      input_arr.append(gts)
#      print(input_arr)
      #print(len(input_arr[0][0][0]))
      #break
      
  input_arr=np.array(input_arr)
  input_arr=np.swapaxes(input_arr,2,3)
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

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample_return():
  label_path='/home/kushagra/RECOLA-Audio-recordings-new/*'
  label_path = Path(label_path)
  audio_paths=[]
  for path in label_path.parent.glob(label_path.name):
      audio_paths.append(path)

  audio_paths.sort()
  audio_all=[]
  print(audio_paths)
  for i in range(len(audio_paths)):
    print(i)
    audio_path=audio_paths[i]
    audio_raw = read_wave(audio_path)
    audio_all.append(audio_raw)
    print('raw_audio shape: ',audio_raw.shape)
  audio_all=np.array(audio_all)
  return audio_all


def serialize_sample(writer,  annotation, upsample=False):
#  classes = [label for _, label in sample_data]
#  class_ids = set(classes)
#  num_samples_per_class = {class_name: sum(x == class_name for x in classes) for class_name in class_ids}
#  print(num_samples_per_class)

#  if upsample:
#    max_samples = np.max(list(num_samples_per_class.values()))
#    augmented_data = []

#    for class_name, n_samples in num_samples_per_class.items():
#        n_samples_to_add = max_samples - n_samples

#        while n_samples_to_add > 0:
#            for sample, label in sample_data:
#                if n_samples_to_add <= 0:
#                    break

#                if label == class_name:
#                    augmented_data.append((sample, label))
#                    n_samples_to_add -= 1

#    print('Augmented the dataset with {} samples'.format(len(augmented_data)))
#    sample_data += augmented_data

#    import random
#    random.shuffle(sample_data)
  label_path='/home/kushagra/RECOLA-Audio-recordings-new/*'
  label_path = Path(label_path)
  audio_paths=[]
  for path in label_path.parent.glob(label_path.name):
      audio_paths.append(path)
  
  audio_paths.sort()
  
  print(audio_paths)
  for i in range(len(audio_paths)-4):
    print(i)
    audio_path=audio_paths[i]
    audio_raw = read_wave(audio_path)
    print('raw_audio_shape: ',audio_raw.shape)
    annotation_raw=annotation[i]
    print('annotation_raw shape: ',annotation_raw.shape)
    #print('audio_raw: ',audio_raw)
    #print('annotation_raw: ',annotation_raw)
    #example = tf.train.Example(features=tf.train.Features(feature={
    #            'label': _int_feauture(label),
    #            'raw_audio': _bytes_feauture(audio.astype(np.float32).tobytes()),
    #        }))
   
    for j in range(577):
	audiowrite=audio_raw[j*13:(j+1)*13,:]
        annotationwrite=annotation_raw[j*13:(j+1)*13,:]
        example = tf.train.Example(features=tf.train.Features(feature={
            'audio_raw': _bytes_feature(audiowrite.astype(np.float32).tobytes()),
            'annotation_raw': _bytes_feature(annotationwrite.astype(np.float32).tobytes())}))
    
        writer.write(example.SerializeToString())
    #break
    #del audio, label

def main(data_folder, tfrecords_folder):

  root_dir = Path(data_folder)
  #features = get_labels(labels_file)
  annotation=get_annotation()
  #for portion in ['train', 'devel']:
  portion='train'
  if not Path(tfrecords_folder).exists():
      Path(tfrecords_folder).mkdir()

  writer = tf.python_io.TFRecordWriter(
      (Path(tfrecords_folder) / '{}audioseq13train19.tfrecords'.format(portion)
  ).as_posix())
  #print('features: ',features)  
  serialize_sample(writer, annotation, upsample='train')
  writer.close()

if __name__ == '__main__':
  main(FLAGS.wave_folder, FLAGS.tf_folder)
