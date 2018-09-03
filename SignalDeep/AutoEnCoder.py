import csv
from keras.layers import Input
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from six.moves import range
import random
from keras import backend as K


with open('Data/TrainingForAuto.csv', 'r', ) as f:
    reader = csv.reader(f)
    rows0 = [row for row in reader]
    column = [row[1] for row in reader]
    a = len(rows0)
rows0 = np.array(rows0)
rows0 = rows0[:, 1:len(rows0[0]) -2 ]
rows0 = rows0.astype(float)

with open('Data/Training.csv', 'r', ) as f:
    reader = csv.reader(f)
    rows1 = [row for row in reader]
    column = [row[1] for row in reader]
    a = len(rows1)
rows1 = np.array(rows1)

with open('Data/Test5.csv', 'r', ) as g:
    reader = csv.reader(g)
    rows2 = [row for row in reader]
    column = [row[1] for row in reader]
    a = len(rows2)
rows2 = np.array(rows2)

X = rows1[:, 1:len(rows1[0]) - 2]
y = rows1[:, len(rows1[0]) - 1]
print(len(X))
X = X.astype(float)
y = y.astype(float)
#y = column_or_1d(y, warn=True)

M = rows2[:, 1:len(rows1[0]) - 2]
M = M.astype(float)

for i in range(len(X) - 1, -1, -1):
    if float(np.max(X[i])) < 50:
        X = np.delete(X, i, axis=0)
        y = np.delete(y, i, axis=0)

print(len(X))

a = list(range(len(X)))
slice = random.sample(a, len(a))
XX1 = X[slice[1]]
yy1 = y[slice[1]]
XX2 = XX1
yy2 = yy1
for j in range(2, len(slice)):
    if j < 1.1 * len(a):
        XX1 = np.row_stack((XX1, X[slice[j]]))
        yy1 = np.row_stack((yy1, y[slice[j]]))
    else:
        XX2 = np.row_stack((XX2, X[slice[j]]))
        yy2 = np.row_stack((yy2, y[slice[j]]))

print(len(rows0), len(rows0[0]))

print(y.dtype)


input_img = Input(shape=(len(rows0[0]),))  # adapt this if using `channels_first` image data format

s = len(rows0[0])

model = Sequential()
model.add(Dense(s, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(1500, activation='softmax'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(s, activation='relu'))

model.compile(optimizer='adam', loss='mse')  # reporting the accuracy
model.fit(rows0, rows0, epochs=10, batch_size=128, shuffle=True, validation_data=(rows0, rows0))

#mid = model(input_img, y)
#reduced_representation0 = model.predict(XX1)
#reduced_representation1 = model.predict(M)


get_14th_layer_output = K.function([model.layers[0].input], [model.layers[14].output])


# define model structure
def baseline_model():
   model = Sequential()
   model.add(Dense(150, input_dim=250, activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(75, activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(25, activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(10, activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(2, activation='softmax'))
   # Compile model
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=15, batch_size=16)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.


m = []
for k in range(5):
    X_train = get_14th_layer_output([XX1])
    X_test = get_14th_layer_output([M])
    Y_train = yy1
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_train = np_utils.to_categorical(Y_train)
    estimator.fit(X_train, Y_train)
    # make predictions
    pred = estimator.predict(X_test)

    print(pred)
    j = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            j = j + 1
    print(j, len(pred))
    m.append(j)
print(sum(m) / len(m), np.std(m))



# Graphing the layers
layer_number = [2, 5, 8, 11, 14, 17, 20, 21, 23]
layer_dat = []
for qlist in layer_number:
    get_nth_layer_output = K.function([model.layers[0].input], [model.layers[qlist].output])
    nlayer = get_nth_layer_output([M])
    layer_dat.append(nlayer[0][0])

layer_node_number = [1950, 1500, 750, 350, 250, 350, 750, 1500, 1950]
mn = []
for qn in layer_node_number:
    qm = []
    for i in range(qn):
        qm.append(i)
    mn.append(qm)


import matplotlib.pyplot as plt

plt.figure(1)

# linear
plt.subplot(331)
plt.plot(mn[0], layer_dat[0])
#plt.yscale('linear')
plt.title('1950')
plt.grid(True)


# log
plt.subplot(332)
plt.plot(mn[1], layer_dat[1])
#plt.yscale('log')
plt.title('1500')
plt.grid(True)


# symmetric log
plt.subplot(333)
plt.plot(mn[2], layer_dat[2])
#plt.plot(x, y - y.mean())
#plt.yscale('symlog', linthreshy=0.01)
plt.title('750')
plt.grid(True)

# logit
plt.subplot(334)
plt.plot(mn[3], layer_dat[3])
#plt.yscale('logit')
plt.title('350')
plt.grid(True)

# linear
plt.subplot(335)
plt.plot(mn[4], layer_dat[4])
#plt.yscale('linear')
plt.title('250')
plt.grid(True)


# log
plt.subplot(336)
plt.plot(mn[5], layer_dat[5])
#plt.yscale('log')
plt.title('350')
plt.grid(True)


# symmetric log
plt.subplot(337)
plt.plot(mn[6], layer_dat[6])
#plt.plot(x, y - y.mean())
#plt.yscale('symlog', linthreshy=0.01)
plt.title('750')
plt.grid(True)

# logit
plt.subplot(338)
plt.plot(mn[7], layer_dat[7])
#plt.yscale('logit')
plt.title('1500')
plt.grid(True)

# logit
plt.subplot(339)
plt.plot(mn[8], layer_dat[8])
#plt.yscale('logit')
plt.title('1950')
plt.grid(True)

plt.show()