from keras.models import Sequential
from keras.layers import BatchNormalization ,Dropout
from keras.layers.convolutional import Conv2D ,MaxPooling2D
from keras.layers.core import  Activation ,Flatten ,Dense
from keras import backend as k

class LeNet:
    @staticmethod
    def build(width,height,depth,calsses):
        model = Sequential()
        inputShape = (height,width,depth)
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(16, (5,5) ,padding="same" ,input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2)))

        model.add(Conv2D(32 , (5,5) ,padding="same"))
        model.add(BatchNormalization())
        model.add((Activation("relu")))
        model.add(MaxPooling2D(pool_size=(2,2) ,strides=(2,2)))

        model.add(Conv2D(64 , (5,5) ,padding="same"))
        model.add(BatchNormalization())
        model.add((Activation("relu")))
        model.add(MaxPooling2D(pool_size=(2,2) ,strides=(2,2)))
        model.add(Dropout(0.15))

        model.add(Conv2D(64 , (5,5) ,padding="same"))
        model.add(BatchNormalization())
        model.add((Activation("relu")))
        model.add(MaxPooling2D(pool_size=(2,2) ,strides=(2,2)))
        model.add(Dropout(0.15))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.15))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.15))

        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.15))

        model.add(Dense(calsses))
        model.add(Activation("softmax"))

        return model