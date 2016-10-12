# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import model_from_json

import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('dorcus_DL_study.csv')

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

# show results
model.summary()
score = model.evaluate(trX, trY, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

# function for generating decision boundary
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = int(trX[:, 0].min()) - 1, int(trX[:, 0].max()) + 1
    y_min, y_max = int(trX[:, 1].min()) - 1, int(trX[:, 1].max()) + 1
    h = 1

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha = 0.4)
    plt.scatter(trX[:,0], trX[:,1], c=Y)

# plot result
plot_decision_boundary(lambda x: model.predict_classes(x))
plt.title('decision_boundary_study')
plt.xlabel('days_elapse')
plt.ylabel('weight_gram')
plt.grid()
plt.legend(loc='lower right')
plt.savefig('decision_boundary_study.png')
plt.show()
