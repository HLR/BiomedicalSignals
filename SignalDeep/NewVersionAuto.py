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
from keras.callbacks import History
from keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.models import model_from_yaml


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

M = rows2[:, 2:len(rows2[0])]
M = M.astype(float)

#for i in range(len(X) - 1, -1, -1):
#    if float(np.max(X[i])) < 50:
#        X = np.delete(X, i, axis=0)
#        y = np.delete(y, i, axis=0)

print(len(X))

a = list(range(len(X)))
slice = random.sample(a, len(a))
XX1 = X[slice[1]]
yy1 = y[slice[1]]
XX2 = XX1
yy2 = yy1
for j in range(1, len(slice)):
    if j < 1.1 * len(a):
        XX1 = np.row_stack((XX1, X[slice[j]]))
        yy1 = np.row_stack((yy1, y[slice[j]]))
    else:
        XX2 = np.row_stack((XX2, X[slice[j]]))
        yy2 = np.row_stack((yy2, y[slice[j]]))

print(len(rows0), len(rows0[0]))

print(len(XX1), len(yy1))


EpochAuto = 100000
DeepEpoch = 1000

#input_img = Input(shape=(len(rows0[0]),))  # adapt this if using `channels_first` image data format

adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)

s = len(rows0[0])

model = Sequential()
model.add(Dense(s, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(1950, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(250, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(750, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(Dense(1950, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(s, kernel_initializer='uniform'))

model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])  # reporting the accuracy
his = model.fit(rows0, rows0, epochs=EpochAuto, batch_size=128, shuffle=True, validation_data=(rows0, rows0))


# serialize model to YAML

model_yaml = model.to_yaml()

with open("model.yaml", "w") as yaml_file:

    yaml_file.write(model_yaml)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")

pred_auto= model.predict([M])

# The number of nodes in each layer
layer_node_number = [1950, 1950, 750, 750, 250, 750, 750, 1950, 1950]
layer_number = [0, 5, 8, 11, 14, 17, 20, 21, 23]
r= 3

get_14th_layer_output = K.function([model.layers[0].input], [model.layers[layer_number[r]].output])




# Defining compression of the layers, in other word the number of each layer
compression = [0.60, 0.50, 0.30, 0.40]
q = []
q.append(layer_node_number[r])
for co in range(len(compression)):
    q.append(int(compression[co] * q[co]))


# define model structure
def baseline_model():
   model = Sequential()
   model.add(Dense(q[1], input_dim=q[0], activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(q[2], activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(q[3], activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(q[4], activation='relu'))
   model.add(Dropout(0.2))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
   model.add(Dense(2, activation='softmax'))
   # Compile model
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model



estimator = KerasClassifier(build_fn=baseline_model, epochs=DeepEpoch, batch_size=16)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.





m = []
LossAuto = []
AccuAuto = []
LossAvModel = []
AccuAvModel = []
for k in range(1):
    X_train = get_14th_layer_output([XX1])
    #X_test = get_14th_layer_output([XX2])
    X_Drug = get_14th_layer_output([M])
    Y_train = yy1
    #Y_test = yy2
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_train = np_utils.to_categorical(Y_train)
    #Y_test = encoder.fit_transform(Y_test)
    #Y_test = np_utils.to_categorical(Y_test)
    hist = estimator.fit(X_train, Y_train, validation_split=0.15)
    LossAvModel.append(hist.history['val_loss'][:][DeepEpoch - 1])
    AccuAvModel.append(hist.history['val_acc'][:][DeepEpoch - 1])
    LossAuto.append(his.history['val_loss'][:][EpochAuto-1])
    AccuAuto.append(his.history['val_acc'][:][EpochAuto-1])
    # make predictions
    pred = estimator.predict(X_Drug)

    print(pred)
    j = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            j = j + 1
    print(j, len(pred))
    m.append(j)
print(sum(m) / len(m), np.std(m))
#print(sum(LossAv) / len(LossAv), np.std(LossAv))
#print(sum(AccuAv) / len(AccuAv), np.std(AccuAv))
print('MinLoss of Model:', min(LossAvModel), 'MaxAcuu of Model:', max(AccuAvModel))
print('MinLoss of AutoEnCoder:', min(LossAuto), 'MaxAcuu of  AutoEnCoder:', max(AccuAuto))


import matplotlib.pyplot as plt
for k in range(len(M)):
    # Graphing the layers
    layer_dat = []
    for qlist in layer_number:
        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[qlist].output])
        nlayer = get_nth_layer_output([M])
        layer_dat.append(nlayer[0][k])


    plt.figure(k)
    for j in range(len(layer_node_number)):
        plt.subplot(330 + j + 1)
        plt.plot(layer_dat[j])
        # plt.yscale('linear')
        plt.title(str(layer_node_number[j]))
        plt.grid(True)
    #plt.show()
    plt.savefig('PlotsTest/Layers_Auto' + str(k))

    plt.figure(k+1)
    plt.subplot(211)
    plt.grid(True)
    plt.plot(M[k])
    plt.title('Real Signal')
    plt.grid(True)
    plt.subplot(212)
    plt.grid(True)
    plt.plot(pred_auto[k])
    plt.title('AutoEnCoder Signal')
    #plt.show()
    plt.savefig('PlotsTest/SignalsR_A'+str(k))



# Accuracy and Loss plots of the AutoEnCoder
plt.figure(83)
plt.subplot(211)
#print(his.history.keys())
# summarize history for accuracy
plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.subplot(212)
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('PlotsTest/Acur_LossOfAuto')
print('Min_Loss:', min(his.history['loss']), 'Max_Accuracy:', max(his.history['val_loss']))




# Accuracy and Loss plots of ML
plt.figure(84)
plt.subplot(211)
#print(his.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.subplot(212)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('PlotsTest/Acur_LossOfML')
