from keras.models import Model,load_model,Sequential
from keras.layers import Input,Dense,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D,Dropout,concatenate,Activation, maximum
from keras.layers import add ,Flatten
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from keras import backend as k
from dataset import batch_test_data , batch_train_data ,cv_imread
import os
import cv2
from keras.preprocessing.image import img_to_array,ImageDataGenerator
from imutils import paths
from keras.applications import Xception ,InceptionV3
import csv
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

seed = 7
np.random.seed(seed)

class typical_network(object):
    def __init__(self):
        self.model = None

        self.class_number = 2

        self.image_width = 2560  # 2560  400 ,1000
        self.image_heiht = 1920  # 1920  300  ,750
        self.depth = 3

        self.class_number = 2

        self.learn_rate = 0.001
        self.Epochs = 60
        self.BatchSize = 8

        self.File_path = "test"
        self.model_path = "modeldir/model.h5"
        self.best_model_path = "lssavemodel/weights.best.hdf5"

    def Conv2d_BN(self,x,nb_filter,kernel_size,strides=(1,1),padding = 'same',name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(nb_filter , kernel_size ,padding=padding ,strides=strides,activation='relu',name=conv_name)(x)
        x = BatchNormalization(axis = 3,name = bn_name)(x)
        return x

    def Conv_Block(self,inpt , nb_filter , kernel_size ,strides=(1,1),with_conv_shortcut=False):
        x = self.Conv2d_BN(inpt,nb_filter=nb_filter[0] , kernel_size=(1,1) , strides=strides,padding = 'same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)
            x = add([x,shortcut])
            return x
        else:
            x = add([x,inpt])
            return x

    def Resnet_50(self):
        inpt = Input(shape=(self.image_heiht, self.image_width, self.depth))
        x = ZeroPadding2D((3, 3))(inpt)
        x = self.Conv2d_BN(x, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = self.Conv_Block(x, nb_filter=[32, 32, 128], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[32, 32, 128], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[32, 32, 128], kernel_size=(3, 3))

        x = self.Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
        x = self.Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
        x = self.Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
        x = self.Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))

        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(1000, activation='relu')(x)

        x = Dense(512, activation='relu')(x)

        x = Dense(256, activation='relu')(x)

        x = Dense(16, activation='relu')(x)

        x = Dense(2, activation='sigmoid')(x)


        self.model = Model(inputs=inpt,outputs=x)
        self.model.summary()

    def vgg16(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(self.image_heiht, self.image_width, 3), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='sigmoid'))
        self.model.summary()

    def Inception(self, x, nb_filter):
        branch1x1 = self.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branch3x3 = self.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch3x3 = self.Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

        branch5x5 = self.Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch5x5 = self.Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = self.Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
        return x

    def googlenet(self):
        inpt = Input(shape=(self.image_heiht, self.image_width, 3))
        # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = self.Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Inception(x, 64)  # 256
        x = self.Inception(x, 120)  # 480
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Inception(x, 128)  # 512
        x = self.Inception(x, 128)
        x = self.Inception(x, 128)
        x = self.Inception(x, 132)  # 528
        x = self.Inception(x, 208)  # 832
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Inception(x, 208)
        x = self.Inception(x, 256)  # 1024
        x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(1000, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(2, activation='sigmoid')(x)
        self.model = Model(inpt, x, name='inception')
        self.model.summary()

    def normal_net(self):
        self.model = Sequential()
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

    def triple_generator(generator):
        while True:
            x, y = generator.next()
            exit()
            yield x, [y, y, y, y]

    def xceptionnet(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            width_shift_range=0.4,
            height_shift_range=0.4,
            rotation_range=90,
            zoom_range=0.7,
            horizontal_flip=True,
            vertical_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            'xuelang11\\',
            target_size=(self.image_heiht, self.image_width),
            batch_size=self.BatchSize,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            'xuelang11\\',
            target_size=(self.image_width , self.image_width ),
            batch_size=self.BatchSize,
            class_mode='categorical')

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                                    cooldown=0, min_lr=0)

        if os.path.exists('model.h5'):
            self.model = load_model('model.h5')
        else:
            input_tensor = Input(shape=(self.image_heiht, self.image_width, 3))
            base_model1 = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
            base_model1 = Model(inputs=[base_model1.input], outputs=[base_model1.get_layer('avg_pool').output],
                                name='xception')

            base_model2 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
            base_model2 = Model(inputs=[base_model2.input], outputs=[base_model2.get_layer('avg_pool').output],
                                name='inceptionv3')

            img1 = Input(shape=(self.image_heiht, self.image_width, 3), name='img_1')
            feature1 = base_model1(img1)
            feature2 = base_model2(img1)
            category_predict1 = Dense(48, activation='relu', name='ctg_out_1')(
                Dropout(0.5)(
                    feature1
                )
            )
            category_predict2 = Dense(48, activation='relu', name='ctg_out_2')(
                Dropout(0.5)(
                    feature2
                )
            )
            category_predict = Dense(48, activation='relu', name='ctg_out')(
                concatenate([feature1, feature2])
            )

            max_category_predict = maximum([category_predict1, category_predict2])
            self.model = Model(inputs=[img1],
                          outputs=[category_predict1, category_predict2, category_predict, max_category_predict])

            plot_model(self.model, to_file='single_model.png')

            for layer in base_model1.layers:
                layer.trainable = False

            for layer in base_model2.layers:
                layer.trainable = False
            self.model.compile(optimizer='nadam',
                          loss={
                              'ctg_out_1': 'categorical_crossentropy',#binary_crossentropy
                              'ctg_out_2': 'categorical_crossentropy',
                              'ctg_out': 'categorical_crossentropy',
                              'maximum_1': 'categorical_crossentropy'
                          },
                          metrics=['accuracy'])
            self.model.fit_generator(self.triple_generator(train_generator),
                                steps_per_epoch=1500 / self.BatchSize + 1,
                                epochs=24,
                                validation_data=self.triple_generator(validation_generator),
                                validation_steps=1500 / self.BatchSize + 1,
                                callbacks=[early_stopping, auto_lr])
            self.model.save('model.h5')
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)
        cur_base_model = self.model.layers[1]
        for layer in cur_base_model.layers[:105]:
            layer.trainable = False
        for layer in cur_base_model.layers[105:]:
            layer.trainable = True

        cur_base_model = self.model.layers[2]
        for layer in cur_base_model.layers[:262]:
            layer.trainable = False
        for layer in cur_base_model.layers[262:]:
            layer.trainable = True

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss={
                          'ctg_out_1': 'categorical_crossentropy',
                          'ctg_out_2': 'categorical_crossentropy',
                          'ctg_out': 'categorical_crossentropy',
                          'maximum_1': 'categorical_crossentropy'
                      },
                      metrics=['accuracy'])
        batch_size = self.BatchSize * 3 / 4
        train_generator = test_datagen.flow_from_directory(
            'xuelang11',
            target_size=(self.image_heiht, self.image_width),
            batch_size=self.BatchSize,
            class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            'xuelang11',
            target_size=(self.image_heiht, self.image_width),
            batch_size=self.BatchSize,
            class_mode='categorical')
        save_model = ModelCheckpoint('lssavemodel\xception-tuned{epoch:02d}-{val_ctg_out_acc:.2f}.h5')
        self.model.fit_generator(self.triple_generator(train_generator),
                            steps_per_epoch=1500 / batch_size + 1,
                            epochs=24,
                            validation_data=self.triple_generator(validation_generator),
                            validation_steps=1500 / batch_size + 1,
                            callbacks=[early_stopping, auto_lr,
                                       save_model])  # otherwise the generator would loop indefinitely
        self.model.save('model_tuned.h5')

    def train(self,trainx, trainy, testx, testy):
        #
        # trainx,trainy = batch_train_data(train_file)
        # testx , testy = batch_test_data(verification_file)
        # load weights
        # self.model.load_weights(self.best_model_path)

        # initialize the model
        print("[info]:compiling model.....")
        opt = Adam(lr=self.learn_rate, decay=1e-6)

        # opt = Adam(lr=self.learn_rate, decay=self.learn_rate / self.Epochs)

        # sgd = SGD(decay=1e-6,momentum = 0.09,lr=self.learn_rate)
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
        '''
        H = self.model.fit_generator(batch_train_data(train_file, batch_size=1),
                                 validation_data=batch_test_data(verification_file, batch_size=1), validation_steps=1,
                                 epochs=self.Epochs, steps_per_epoch=2, callbacks=callbacks_list)
        '''
        self.model.fit(trainx, trainy, validation_data=(testx, testy), shuffle=True, epochs=self.Epochs,
                   batch_size=self.BatchSize, callbacks=callbacks_list)

    def save(self):
        print("Model saved")
        self.model.save(self.model_path)

    def evaluate_model(self,verification_file):
        print('\nTesting---------------')
        testx,testy= batch_test_data(verification_file)
        loss, accuracy = self.model.evaluate(testx, testy)
        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def load(self):
        print("model load")
        self.model = load_model(self.model_path)

    def best_predict(self):
         # load weights
         self.model.load_weights(self.best_model_path)
         # initialize the model
         print("[info]:compiling model.....")
         sgd = SGD(decay=1e-6, momentum=0.5, lr=self.learn_rate)
         self.model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])  # categorical_crossentropy
         self.model.save(self.model_path)

    def predict(self):
         self.model = load_model(self.model_path)
         csvFile = open("submit.csv", "w" , newline='')
         writer = csv.writer(csvFile)
         list_head =['filename','probability']
         writer.writerow(list_head)
         file_num = list(paths.list_images(self.File_path))
         for each_file in file_num:
             image = cv_imread(each_file)
             image = cv2.resize(image,(1000,750))
             image = img_to_array(image)
             image_arr = np.array(image, dtype="float32") / 255.0
             #image_arr = image_arr.reshape((1, 1920, 2560, 3))

             image_arr = image_arr.reshape((1, 750, 1000, 3))

             result = self.model.predict(image_arr)
             probility = round(result[0][1] ,6)
             each_file = each_file.replace('\ ', ' ')
             each_file = os.path.basename(each_file)
             add_info = [each_file, probility]
             writer.writerow(add_info)
         csvFile.close()








