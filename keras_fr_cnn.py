#coding:utf-8
#python -m pip install h5py
from __future__ import print_function  
import numpy as np
np.random.seed(1337)  # for reproducibility
  
from PIL import Image  

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD  
from keras.utils import np_utils
from keras import backend as K

'''
Olivetti Faces是纽约大学的一个比较小的人脸库，
由40个人的400张图片构成，即每个人的人脸图片为10张。
每张图片的灰度级为8位，每个像素的灰度大小位于0-255之间。
整张图片大小是1190 × 942，一共有20 × 20张照片。
那么每张照片的大小就是（1190 / 20）× （942 / 20）= 57 × 47 。
'''

# There are 40 different classes  
nb_classes = 40  # 40个类别
epochs = 40  # 进行40轮次训
batch_size = 40  # 每次迭代训练使用40个样本
  
# input image dimensions  
img_rows, img_cols = 57, 47  
# number of convolutional filters to use  
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
# size of pooling area for max pooling  
nb_pool = 2  
# convolution kernel size  
nb_conv = 3  # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
  
def load_data(dataset_path):  
    img = Image.open(dataset_path)  
    img_ndarray = np.asarray(img, dtype = 'float64') / 255  # asarray，将数据转化为np.ndarray，但使用原内存
    # 400 pictures, size: 57*47 = 2679  
    faces = np.empty((400, 2679)) 
    for row in range(20):  
        for column in range(20):
           faces[row * 20 + column] = np.ndarray.flatten(img_ndarray[row*57 : (row+1)*57, column*47 : (column+1)*47]) 
           # flatten将多维数组降为一维
  
    label = np.empty(400)  
    for i in range(40):
        label[i*10 : i*10+10] = i  
    label = label.astype(np.int)  
  
    #train:320,valid:40,test:40  
    train_data = np.empty((320, 2679))  
    train_label = np.empty(320)  
    valid_data = np.empty((40, 2679))  
    valid_label = np.empty(40)  
    test_data = np.empty((40, 2679))  
    test_label = np.empty(40)  
  
    for i in range(40):
        train_data[i*8 : i*8+8] = faces[i*10 : i*10+8] # 训练集中的数据
        train_label[i*8 : i*8+8] = label[i*10 : i*10+8]  # 训练集对应的标签
        valid_data[i] = faces[i*10+8] # 验证集中的数据
        valid_label[i] = label[i*10+8] # 验证集对应的标签
        test_data[i] = faces[i*10+9] 
        test_label[i] = label[i*10+9]   
    
    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')
       
    rval = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]  
    return rval  
  
def set_model(lr=0.005,decay=1e-6,momentum=0.9): 
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape = (1, img_rows, img_cols)))
    else:
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape = (img_rows, img_cols, 1)))
    model.add(Activation('tanh'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Conv2D(10, kernel_size=(3, 3)))  
    model.add(Activation('tanh'))  
    #model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    model.add(Flatten())      
    model.add(Dense(1000)) #Full connection  
    model.add(Activation('tanh')) 
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))  
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)  
    return model  
  
def train_model(model,X_train, Y_train, X_val, Y_val):  
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,  
          verbose=1, validation_data=(X_val, Y_val))  
    model.save_weights('model_weights.h5', overwrite=True)  
    return model  
  
def test_model(model,X,Y):  
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X, Y, verbose=0)
    return score  
  
if __name__ == '__main__':  
    # the data, shuffled and split between tran and test sets  
    (X_train, y_train), (X_val, y_val),(X_test, y_test) = load_data('olivettifaces.gif')  
    
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)  
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)  
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
        input_shape = (img_rows, img_cols, 1) # 1 为图像像素深度
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples') 
    print(X_val.shape[0], 'validate samples')  
    print(X_test.shape[0], 'test samples')
  
    # convert class vectors to binary class matrices  
    Y_train = np_utils.to_categorical(y_train, nb_classes)  
    Y_val = np_utils.to_categorical(y_val, nb_classes)  
    Y_test = np_utils.to_categorical(y_test, nb_classes)  
  
    model = set_model()
    train_model(model, X_train, Y_train, X_val, Y_val)   
    score = test_model(model, X_test, Y_test)  
  
    model.load_weights('model_weights.h5')  
    classes = model.predict_classes(X_test, verbose=0)  
    test_accuracy = np.mean(np.equal(y_test, classes))  
    print("accuarcy:", test_accuracy)
    for i in range(0,40):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i]);
    