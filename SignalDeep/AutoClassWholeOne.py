import scipy
from scipy.stats import pearsonr
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape, concatenate
from keras.models import Sequential, Model
import keras
from keras import backend as K


from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format




shared_encoder = Sequential()
#shared_encoder.add(Input(a))
shared_encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
shared_encoder.add(MaxPooling2D((2, 2)))
shared_encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
shared_encoder.add(MaxPooling2D((2, 2)))
shared_encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
shared_encoder.add(MaxPooling2D((2, 2)))
shared_encoder.add(Flatten())
shared_encoder.add(Dense(128, activation='relu'))
shared_encoder.add(Dropout(0.5))



# at this point the representation is (4, 4, 8) i.e. 128-dimensional

shared_decoder = Sequential()
shared_decoder.add(Dense(128, activation='relu'))
shared_decoder.add(Dropout(0.5))
shared_decoder.add(Reshape((4, 4, 8), input_shape=(128,)))
shared_decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
shared_decoder.add(UpSampling2D((2, 2)))
shared_decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
shared_decoder.add(UpSampling2D((2, 2)))
shared_decoder.add(Conv2D(16, (3, 3), activation='relu'))
shared_decoder.add(UpSampling2D((2, 2)))
shared_decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='EnCode_Classifier_Out'))



#model_ClassAuto = Sequential()
a = Input(shape=(28, 28, 1), name = 'main_input')
model_ClassAuto = shared_encoder(a)
ClassifierEn = Dense(10, activation="softmax", name= 'ClassifierEn')(model_ClassAuto)
EnCode_Classifier_Out = shared_decoder(ClassifierEn)
model_ClassAuto = Model(inputs=a, outputs=EnCode_Classifier_Out)



model_ClassAuto.compile(optimizer='rmsprop', loss= 'mse')
model_ClassAuto.fit(x_train,  x_train, epochs=10000, batch_size=128)
testmodel = model_ClassAuto.evaluate(x_test, x_test, verbose=1)

print(testmodel)


model_ClassAuto_yaml = model_ClassAuto.to_yaml()

with open("model_ClassAuto.yaml", "w") as yaml_file:

    yaml_file.write(model_ClassAuto_yaml)

# serialize weights to HDF5

model_ClassAuto.save_weights("model_ClassAuto.h5")

print("Saved model_ClassAuto to disk")

