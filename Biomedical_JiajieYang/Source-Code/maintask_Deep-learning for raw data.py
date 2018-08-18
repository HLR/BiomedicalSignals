import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pylab as pl
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib
from sklearn.decomposition import PCA


# load dataset
with open('E:\python\maintask\maintask7\Training.csv', 'r', ) as f:
	reader = csv.reader(f)
	rows1 = [row for row in reader]
	column = [row[1] for row in reader]
	a=len(rows1)
rows1=np.array(rows1)

with open('E:\python\maintask\maintask7\Test5.csv', 'r', ) as g:
	reader = csv.reader(g)
	rows2 = [row for row in reader]
	column = [row[1] for row in reader]
	a=len(rows2)
rows2=np.array(rows2)

X=rows1[:,1:len(rows1[0])-2]
y=rows1[:,len(rows1[0])-1]
print(len(X))
X=X.astype(float)
y=y.astype(float)

M=rows2[:,1:len(rows1[0])-2]
M=M.astype(float)

for i in range(len(X)-1,-1,-1):
	if float(np.max(X[i])) < 50:
		X=np.delete(X,i,axis=0)
		y=np.delete(y,i,axis=0)

print(len(X))

a=list(range(len(X)))
slice = random.sample(a, len(a))
XX1=X[slice[1]]
yy1=y[slice[1]]
XX2=XX1
yy2=yy1
for j in range(2,len(slice)):
	if j<1.1*len(a):
		XX1=np.row_stack((XX1, X[slice[j]]))
		yy1=np.row_stack((yy1, y[slice[j]]))
	else:
		XX2=np.row_stack((XX2, X[slice[j]]))
		yy2=np.row_stack((yy2, y[slice[j]]))

# define model structure
def baseline_model():
	model = Sequential()
	model.add(Dense(output_dim=3000, input_dim=1950, activation='relu'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dense(output_dim=700, input_dim=3000, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dense(output_dim=200, input_dim=700, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dense(output_dim=50, input_dim=200, activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	model.add(Dense(output_dim=2, input_dim=50, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=32)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train=X
X_test=M
Y_train=y


encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_train = np_utils.to_categorical(Y_train)
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)

print(pred)
j=0
for i in range(len(pred)):
	if pred[i] == 0:
		j=j+1
print(j, len(pred))


