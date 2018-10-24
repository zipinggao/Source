from keras.models import Sequential
from keras.layers.convolutional import Conv2D ,MaxPooling2D
from keras.layers.core import  Activation ,Flatten ,Dense
from keras import backend as k
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

class LeNet:
    @staticmethod
    def build(width,height,depth,calsses):
        model = Sequential()
        inputShape = (height,width,depth)
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(
            Convolution2D(
                filters=2,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='tf',
                input_shape=inputShape,

            )
        )
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        model.add(Flatten())
        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))


        model.add(Dense(calsses))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        model.summary()

        return model
