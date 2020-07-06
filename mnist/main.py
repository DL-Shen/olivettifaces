#coding:utf-8
import numpy as np

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.datasets import mnist

epochs = 40
batch_size = 100


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train= np.asarray(x_train, dtype='float32')
    x_test = np.asarray(x_test,dtype='float32')
    x_train = x_train/255.0
    x_test = x_test/255.0
    x_train = x_train.reshape(x_train.shape[0], 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28,28,1)

    y_train = np_utils.to_categorical(y_train,10)
    y_test= np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def set_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(2,2), input_shape=(28,28,1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(8, kernel_size=(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01,momentum=0.9,nesterov=True)
    model.compile(sgd,loss=keras.losses.categorical_crossentropy)
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save('mnist.h5')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    # model = set_model()
    # model = train_model(model, x_train, y_train)

    model = load_model('mnist.h5')
    print(model.layers[0].input_shape)
    print(model.summary())
    (_, _), (_, y_test_ori) = mnist.load_data()
    classes = model.predict_classes(x_test)  # 输出的是最终分类结果，如5、7
    acc = np.mean(np.equal(classes, y_test_ori))
    print(acc)

    res = model.predict(x_test)  # 输出的是概率
    print(np.argmax(res[0]), y_test_ori[0])
