from keras.applications import Xception,VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout
from keras.models import Model, load_model,Sequential
from keras.optimizers import SGD
from keras import Input
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
image_height= 960 #
image_width= 1280 #
channels= 3
batch_size = 8
def set_npy():
    # model = VGG16(weights=None, include_top=False,
    #                           input_shape=(image_height,image_width,channels),
    #                           classes=1024, pooling='max')

    model = VGG16(weights=None, include_top=False)
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
     )
    test_datagen = ImageDataGenerator(rescale = 1./255)
    generator =  train_datagen.flow_from_directory(
            'twoaddimage/traindata',
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    bottleneck_features_train = model.predict_generator(generator,4395)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    generator =  train_datagen.flow_from_directory(
            'twoaddimage/valdata',
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator,105)
    np.save('bottleneck_features_test.npy', bottleneck_features_validation)

def train():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0]*1259+[1]*658)

    validation_data = np.load('bottleneck_features_train.npy')
    validation_labels = np.array([0]*57+[1]*48)
    #
    # inputx = Input(shape=train_data[0][1:0])
    # x = Flatten()(np.array(train_data))
    x = Dense(256, activation='relu')(train_data)
    x = Dropout(0.5)(x)
    predict = Dense(1, activation='sigmoid')(x)
    model = Model(train_data, predict, name='xception')

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data ,train_labels,
              nb_epoch=50,batch_size=24,
              validation_data=(validation_data,validation_labels))
    model.save('xecombine.h5')
set_npy()
# train()
