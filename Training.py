# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('dorcus_DL_study.csv')

# number of training
numTraining = 80000

# number of class
Cls = list(df['class'].unique()) # [1, 2, 3, 4]
numCls = len(Cls)

# set data
Y = df['class']
X = df[['days_elapse', 'weight_gram']]

# convert data into vector
def __trY(y):
    for i, t in enumerate(y):
        yield np.eye(1, numCls, t-1)

# set data for training
trY = np.array(list(__trY(Y))).reshape(len(Y), numCls)
trX = np.array(X)

# set model
model = Sequential()
model.add(Dense(output_dim=3, input_dim=2, init='normal'))
model.add(Activation('relu'))
model.add(Dense(output_dim=4, input_dim=3, init='normal'))
model.add(Activation('softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# training
his = model.fit(trX, trY, nb_epoch=numTraining)

# plot result
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(his.history['loss'])

# save figure
plt.savefig('loss.png')

# save model
json_string = model.to_json()
open('model.json', 'w').write(json_string)

# save parameters
model.save_weights('param.h5')
