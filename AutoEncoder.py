# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('dorcus_DL_study.csv')

# number of training
numTraining = 100

# number of class
Cls = list(df['class'].unique()) # [1, 2, 3, 4]
numCls = len(Cls)

# set data
X = df[['days_elapse', 'weight_gram']]

# set data for training
trX = np.array(X)

# encode
autoencoder = Sequential()
autoencoder.add(Dense(output_dim=2, input_dim=2, init='uniform'))
autoencoder.add(Activation('relu'))
autoencoder.add(Dense(output_dim=2, input_dim=2, init='uniform'))
autoencoder.add(Activation('relu'))
autoencoder.add(Dense(output_dim=4, input_dim=2, init='uniform'))
autoencoder.add(Activation('softmax'))

# decode
autoencoder.add(Dense(output_dim=2, input_dim=4, init='uniform'))
autoencoder.add(Activation('softmax'))
autoencoder.add(Dense(output_dim=10, input_dim=10, init='uniform'))
autoencoder.add(Activation('relu'))
autoencoder.add(Dense(output_dim=2, input_dim=10, init='uniform'))
autoencoder.add(Activation('relu'))

# compile
# autoencoder.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
autoencoder.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# training
his = autoencoder.fit(trX, trX, nb_epoch=numTraining, verbose=2)

# plot result
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(his.history['loss'])

# save figure
plt.savefig('loss_enc.png')

# save model
json_string = autoencoder.to_json()
open('model_enc.json', 'w').write(json_string)

# save parameters
autoencoder.save_weights('param_enc.h5')
