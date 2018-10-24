from keras.applications import Xception,VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout
from keras.models import Model, load_model,Sequential
from keras.optimizers import SGD
from keras import Input
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
image_height= 750 #
image_width= 1000 #
channels= 3
batch_size = 8
File_path = 'test11\\test'
def set_npy():
    # model = Xception(weights=None, include_top=False,
    #                           input_shape=(image_height,image_width,channels),
    #                           classes=1024, pooling='max')

    model = Xception(weights='imagenet', include_top=False)
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
     )
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator =  train_datagen.flow_from_directory(
            'twoclassoigia_com/traindata',
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(train_generator)
    np.save('bottleneck_features_train2.npy', bottleneck_features_train)
    test_generator =  test_datagen.flow_from_directory(
            'twoclassoigia_com/valdata',
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(test_generator)
    np.save('bottleneck_features_test2.npy', bottleneck_features_validation)

def train():
    train_data = np.load('bottleneck_features_train2.npy')
    train_labels = np.array([0]*1615+[1]*1677)

    validation_data = np.load('bottleneck_features_test2.npy')
    validation_labels = np.array([0]*69+[1]*65)
    #
    print(train_data.shape)
    print(validation_data.shape)

    # inputx = Input(shape=train_data)

    # x = Flatten()(np.array(train_data))
    # img_tensor = tf.convert_to_tensor(train_data)
    # x = Dense(256, activation='relu')(img_tensor)
    # x = Dropout(0.5)(x)

    # predict = Dense(1, activation='sigmoid')(x)
    # model = Model(img_tensor, predict, name='xception')

    # img_tensor = tf.convert_to_tensor(train_data)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256,activation='relu'))
    # model.add(Dense(256,activation='relu',input_dim = 2048))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data ,train_labels,
              nb_epoch=50,batch_size=24,
              validation_data=(validation_data,validation_labels))
    model.save('xecombine.h5')

def predict():
    # model = Xception(weights='imagenet', include_top=False)
    # test_datagen = ImageDataGenerator(rescale = 1)
    # test_generator =  test_datagen.flow_from_directory(
    #         'test11',
    #         target_size=(image_height, image_width),
    #         batch_size=batch_size,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_validation = model.predict_generator(test_generator)
    # np.save('bottleneck_features_testover.npy', bottleneck_features_validation)
    model = load_model('xecombine.h5')
    test_data = np.load('bottleneck_features_testover.npy')
    print(test_data.shape)
    result = model.predict(test_data)
    print(result)
    print(len(result))


set_npy()
# train()
predict()