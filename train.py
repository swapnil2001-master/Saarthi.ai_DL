# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub
import datetime, os
import pandas as pd
import numpy as np
import argparse
import sys
import json

f = open(sys.argv[2],)
print(sys.argv[2])
config = json.load(f)

EPOCHS = config['Epochs']
Data_Path = config['Data_Path']
Val_Data_Path = config['Val_Data_Path']
Test_Data_Path = config['Test_Data_Path']
Batch_size = config['Batch_size']


#print(EPOCHS,Data_Path,Val_Data_Path)

"""# GPU Setup"""

print("Number of GPU's:",len(tf.config.list_physical_devices('GPU')))

data  = pd.read_csv(Data_Path)
val_data = pd.read_csv(Val_Data_Path)
test_data = pd.read_csv(Test_Data_Path)

action_labels = list(set(data['action'])) # 6
object_labels = list(set(data['object'])) # 14
location_labels = list(set(data['location'])) # 4
a_map = {action_labels[i] : i for i in range(len(action_labels))}
o_map = {object_labels[i] : i for i in range(len(object_labels))}
l_map = {location_labels[i] : i for i in range(len(location_labels))}

def prepare_data(data):
  X = list(data['transcription'])
  Y1 = list(data['action'])
  Y2 = list(data['object'])
  Y3 = list(data['location'])
  Y1 = np.array([a_map[x] for x in Y1])
  Y2 = np.array([o_map[x] for x in Y2])
  Y3 = np.array([l_map[x] for x in Y3])
  return np.array(X),(Y1,Y2,Y3)

X,Y = prepare_data(data)
val_X,val_Y = prepare_data(val_data)
test_X,test_Y = prepare_data(test_data)

"""Model"""

base_layer_1 = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',trainable = False)

inputs = tf.keras.layers.Input(shape = [], dtype = tf.string , name = 'Transcription')

base_model_layer = base_layer_1(inputs)
lam_layer = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis =1))(base_model_layer)
x = tf.keras.layers.LSTM(256)(lam_layer)
x = tf.keras.layers.Dense(128, activation= 'relu')(x)


o1 = tf.keras.layers.Dense(6,activation='softmax',name='Action')(x)
o2 = tf.keras.layers.Dense(14,activation='softmax',name='Object')(x)
o3 = tf.keras.layers.Dense(4,activation='softmax',name = 'Location')(x)
out = [o1,o2,o3]

conv_model = tf.keras.Model(inputs, out)
conv_model.summary()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

conv_model.compile(loss =tf.keras.losses.sparse_categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adam(),
              metrics = 'accuracy')

"""Training"""

conv_model.fit(x=X,y =Y,
                validation_data=(val_X,val_Y),
                epochs = EPOCHS,
                batch_size=Batch_size,
                callbacks=[tensorboard_callback])





#Calculating the F1 score

from sklearn.metrics import f1_score
test_pred = conv_model.predict(test_X)
tlabel1 = np.argmax(test_pred[0],axis=1)
tlabel2 = np.argmax(test_pred[1],axis=1)
tlabel3 = np.argmax(test_pred[2],axis=1)

f1_action = f1_score(test_Y[0], tlabel1, average='macro')
f1_object = f1_score(test_Y[1], tlabel2, average='macro')
f1_location = f1_score(test_Y[2], tlabel3, average='macro')

f1_avg = f1_action+f1_location+f1_object
f1_avg = f1_avg / 3.0
print(f1_avg)
