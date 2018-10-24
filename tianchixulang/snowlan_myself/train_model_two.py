from keras.models import Sequential,load_model
from keras.utils.training_utils import multi_gpu_model
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras import backend as k
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam
from dataset import batch_test_data , batch_train_data ,cv_imread
from keras.preprocessing.image import img_to_array,ImageDataGenerator
import numpy as np
import os
from imutils import paths
import csv
import cv2

class Model(object):
     def __init__(self):
         self.model = None

         self.class_number = 2

         self.image_width = 1000  #2560  400  600  1000
         self.image_heiht = 750  #1920  300  450  750
         self.depth = 3

         self.class_number = 2

         self.learn_rate = 1e-3
         self.Epochs = 50
         self.BatchSize = 32

         self.File_path ="test"
         self.model_path = "modeldir/model.h5"

     def build_model(self):
         self.model= Sequential()
         inputShape = (self.image_heiht, self.image_width, self.depth)
         if k.image_data_format() == "channels_first":
            inputShape = (self.depth, self.image_heiht, self.image_width)

         self.model.add(Conv2D(4, (3, 3), padding="same", input_shape=inputShape))
         self.model.add(BatchNormalization())
         self.model.add(Activation("relu"))
         self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

         self.model.add(Conv2D(32, (3, 3), padding="same"))
         self.model.add(BatchNormalization())
         self.model.add((Activation("relu")))
         self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

         self.model.add(Conv2D(64, (3, 3), padding="same"))
         self.model.add(BatchNormalization())
         self.model.add((Activation("relu")))
         self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
         self.model.add(Dropout(0.15))

         self.model.add(Conv2D(64, (3, 3), padding="same"))
         self.model.add(BatchNormalization())
         self.model.add((Activation("relu")))
         self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
         self.model.add(Dropout(0.15))

         self.model.add(Flatten())
         self.model.add(Dense(64))
         self.model.add(BatchNormalization())
         self.model.add(Activation("relu"))
         self.model.add(Dropout(0.15))

         self.model.add(Dense(32))
         self.model.add(BatchNormalization())
         self.model.add(Activation("relu"))
         self.model.add(Dropout(0.15))

         self.model.add(Dense(16))
         self.model.add(BatchNormalization())
         self.model.add(Activation("relu"))
         self.model.add(Dropout(0.45))

         self.model.add(Dense(self.class_number))
         self.model.add(BatchNormalization())
         self.model.add(Activation("sigmoid"))
         self.model.summary()


     def build_model_info(self):
         self.model = Sequential()
         inputShape = (self.image_heiht, self.image_width, self.depth)
         if k.image_data_format() == "channels_first":
             inputShape = (self.depth, self.image_heiht, self.image_width)

         self.model.add(
             Convolution2D(
                 filters=2,
                 kernel_size=(5, 5),
                 padding='same',
                 dim_ordering='tf',
                 input_shape=inputShape,

             )
         )

         self.model.add(BatchNormalization())
         self.model.add(Activation('relu'))
         self.model.add(
             MaxPooling2D(
                 pool_size=(2, 2),
                 strides=(2, 2),
                 padding='same'
             )
         )

         self.model.add(Flatten())
         self.model.add(Dense(16))
         self.model.add(BatchNormalization())
         self.model.add(Activation('relu'))
         self.model.add(Dropout(0.15))

         self.model.add(Dense(self.class_number))
         self.model.add(BatchNormalization())
         self.model.add(Activation('sigmoid'))
         self.model.summary()
         #return model

     def train(self,train_file,verification_file):

         trainx,trainy=batch_train_data(train_file)


         '''
         aug = ImageDataGenerator(rotation_range = 20 , width_shift_range = 0.1 ,height_shift_range = 0.1,shear_range = 0.2,
                                  zoom_range = 0.2,horizontal_flip =True , fill_mode ="nearest")
        '''
         # load weights
         #self.model.load_weights('weights.best.hdf5')

         # initialize the model
         print("[info]:compiling model.....")
         opt = Adam(lr=self.learn_rate, decay=self.learn_rate / self.Epochs)
         self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) #categorical_crossentropy


         #ckeckpoint model  效果有提升便保存
         '''
         Lsfile_path = 'lssavemodel\weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
         checkpoint = ModelCheckpoint(Lsfile_path ,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
         callbacks_list = [checkpoint]
         '''
         #保存最好的一次
         filepath = 'lssavemodel\weights.best.hdf5'
         checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
         callbacks_list = [checkpoint]

         # train the network
         print("[info]:training model.....")
         H = self.model.fit_generator(trainx,trainy,batch_size=32,
                                 validation_data=(testx,texty),
                                 epochs=self.Epochs, callbacks=callbacks_list)
         #aug.flow(trainx,trainy,self.BatchSize), steps_per_epoch=1500,
         #validation_data=batch_test_data(verification_file, batch_size=32), validation_steps=1,

     def save(self):
         print("Model saved")
         self.model.save(self.model_path)

     def evaluate_model(self,verification_file):
         print('\nTesting---------------')
         testx,texty=batch_test_data(verification_file)
         loss, accuracy = self.model.evaluate(testx, texty)
         print('test loss;', loss)
         print('test accuracy:', accuracy)

     def load(self):
         print("model load")
         self.model = load_model(self.model_path)

     def predict(self):
         self.model = load_model(self.model_path)
         csvFile = open("submit.csv", "w" , newline='')
         writer = csv.writer(csvFile)
         list_head =['filename','probability']
         writer.writerow(list_head)
         file_num = list(paths.list_images(self.File_path))
         i = 1
         for each_file in file_num:
             image = cv_imread(each_file)
             image = img_to_array(image)
             image_arr = np.array(image, dtype="float32") / 255.0
             image_arr = image_arr.reshape((1, 1920, 2560, 3))
             result = self.model.predict_proba(image_arr)
             probility = round(result[0][1] ,5)
             each_file = each_file.replace('\ ', ' ')
             each_file = os.path.basename(each_file)
             add_info = [each_file, probility]
             writer.writerow(add_info)
         csvFile.close()






