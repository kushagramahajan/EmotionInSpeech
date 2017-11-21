import numpy as np
import sklearn
from sklearn.svm import SVR
from pathlib import Path
import pickle
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor

__signal_framerate = 16000
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
              if(counter%7502==0):
                  continue
              l=l[:-1]
              #l.replace("\n","")
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
  input_arr=np.reshape(input_arr,(172477,6))
  #print(input_arr[0])
  print('annotation shape: ',input_arr.shape)
  return input_arr




def get_labels():
  """Parses the data arff files to extract the labels 

  Args:
      label_path: A path glob which contains the arff files with the labels.
  Returns:
      A dictionary for the labels of each fold.
  """
  labels = {}
  label_path='/home/kushagra/RECOLA-Audio-features/*'
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
              #l.replace("\r","")
              #l.replace("\n","")

              if(l[0]!='@' and len(l)>2):
                  #print(l.split(',')[1:])
                  l=l[:-2]
                  gts.append([l.split(',')[1:]])
                  #break
#      gts=np.array(gts)
      
      input_arr.append(gts)
      #print(gts[-1])
#      print(input_arr)
      #print(len(input_arr[0][0][0]))
      #break

  input_arr=np.array(input_arr)
#  print(input_arr.shape)
  input_arr=np.swapaxes(input_arr,2,3)
  input_arr=np.squeeze(input_arr, axis=3)
  input_arr=np.reshape(input_arr,(172477,130))
  #print(input_arr[0])
  print('label shape: ',input_arr.shape)
  return input_arr

for in range(len(labels[0])):
  labels[:,i]=(labels[:,i]-np.mean(labels[:,i]))/float(np.std(labels[:,i]))


for i in range(len(annotations[0])):
  annotations[:,i]=(annotations[:,i]-np.min(annotations[:,i]))/float(np.max(annotations[:,i])-np.min(annotations[:,i]))


def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

annotations=get_annotation()
labels=get_labels()
#for i in range(len(annotations)):
#  for j in range(len(annotations[0])):
#    annotations[i][j]=float(annotations[i][j])


#for i in range(len(labels)):
#  for j in range(len(labels[0])):
#    labels[i][j]=float(labels[i][j])

annotations = annotations.astype(np.float)
labels = labels.astype(np.float)

#print('tst print label')
#print(labels[142481])

#print('tst print annotation')
#print(annotations[142481])
Xtrain=labels[:142481,:]
Ytrain=annotations[:142481,:]

Xtrain=np.array(Xtrain)
print(Xtrain.shape)
Ytrain=np.array(Ytrain)
print(Ytrain.shape)

Xtest=labels[142481:,:]
Ytest=annotations[142481:,:]


Xtest=np.array(Xtest)
print(Xtest.shape)

Ytest=np.array(Ytest)
print(Ytest.shape)

print(labels[142481])
#for i in range(6):  
#clf.fit(Xtrain, Ytrain)
print(annotations[142481])
print(labels[142481][0].dtype)
print(annotations[142482][0].dtype)


#clf = MultiOutputRegressor(SVR(kernel='linear',C=1.0, epsilon=0.2, max_iter=30000), n_jobs=-4)
clf1 = joblib.load('clf1_10l.pkl')
clf2 = joblib.load('clf2_10l.pkl')
clf3 = joblib.load('clf3_10l.pkl')
clf4 = joblib.load('clf4_10l.pkl')
clf5 = joblib.load('clf5_10l.pkl')
clf6 = joblib.load('clf6_10l.pkl')

pred1=clf1.predict(Xtest)
pred2=clf2.predict(Xtest)
pred3=clf3.predict(Xtest)
pred4=clf4.predict(Xtest)
pred5=clf5.predict(Xtest)
pred6=clf6.predict(Xtest)

print(pred1.shape)
#print('score: ',clf.score(Xtrain,Ytrain))

u= ((Ytest[:,0] - pred1) ** 2).sum()
v= ((Ytest[:,0] - Ytest[:,0].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)


u= ((Ytest[:,1] - pred2) ** 2).sum()
v= ((Ytest[:,1] - Ytest[:,1].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)


u= ((Ytest[:,2] - pred3) ** 2).sum()
v= ((Ytest[:,2] - Ytest[:,2].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)


u= ((Ytest[:,3] - pred4) ** 2).sum()
v= ((Ytest[:,3] - Ytest[:,3].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)


u= ((Ytest[:,4] - pred5) ** 2).sum()
v= ((Ytest[:,4] - Ytest[:,4].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)


u= ((Ytest[:,5] - pred6) ** 2).sum()
v= ((Ytest[:,5] - Ytest[:,5].mean()) ** 2).sum()
score=1-(u/v)
print('u: ',u)
print('v: ',v)
print('score: ',score)
