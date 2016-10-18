# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import model_from_json

import matplotlib.pyplot as plt

"""
Module for additional training
"""

# load dataset
df = pd.read_csv('dorcus_DL_study.csv')

# number of training
numTraining = 10000

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

# load model
model = model_from_json(open('model.json').read())

# load parameters
model.load_weights('param.h5')

# compile
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# training
his = model.fit(trX, trY, nb_epoch=numTraining, verbose=2)

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
