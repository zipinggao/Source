import os
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout
from keras.models import Model, load_model
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array
import matplotlib.pyplot as plt
from keras.applications  import Xception
import tensorflow as tf
from imutils import paths
import cv2
import math
import csv
import numpy as np
from dataset import cv_imread

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

class PowerTransferMode:
    # 数据准备
    def DataGen(self, dir_path, img_row, img_col, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         zoom_range=0.25, rotation_range=15.,
                                         channel_shift_range=25., width_shift_range=0.02, height_shift_range=0.02,
                                         horizontal_flip=True, fill_mode='constant')
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            dir_path, target_size=(img_row, img_col),
            batch_size=batch_size,
            # class_mode='binary',
            shuffle=is_train)

        return generator

    # ResNet模型
    def ResNet50_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                       is_plot_model=False):
        color = 3 if RGB else 1
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                              input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘制模型
        # if is_plot_model:
        #     # plot_model(model, to_file='resnet50_model.png', show_shapes=True)

        return model

    # VGG模型
    def VGG19_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                    is_plot_model=False):
        color = 3 if RGB else 1
        base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                           classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        # if is_plot_model:
        #     plot_model(model, to_file='vgg19_model.png', show_shapes=True)

        return model

    # InceptionV3模型
    def InceptionV3_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                          is_plot_model=False):
        color = 3 if RGB else 1
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                                 input_shape=(img_rows, img_cols, color),
                                 classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        # if is_plot_model:
        #     # plot_model(model, to_file='inception_v3_model.png', show_shapes=True)

        return model

    #Xception模型
    def Xception_model(self, lr=0.001, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=1000, img_cols=750, RGB=True,
                          is_plot_model=False):
        color = 3 if RGB else 1
        base_model = Xception(weights='imagenet', include_top=False,
                                 input_shape=(img_cols,img_rows,color),
                                 classes=1024,pooling='max')
        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        # 添加自己的全链接分类层
        # x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.4)(x)
        predictions = Dense(nb_classes, activation='sigmoid')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        # sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        # opt = Adam(lr=0.001, decay=1e-6)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#binary_crossentropy

        # 绘图
        # if is_plot_model:
        #     # plot_model(model, to_file='Xception_model.png', show_shapes=True)

        return model

    # 训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps,
                    model_url, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # 模型保存
        model.save(model_url, overwrite=True)
        return history_ft

    # 画图
    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'b-')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'b-')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()

    def dividepredict(self):
        model = load_model("sing_xception.h5")
        image_path = "test11\\test"
        width = 600
        height = 450
        divwidth = math.ceil(2560 / width)
        divheight = math.ceil(1920 / height)
        file_num = list(paths.list_images(image_path))
        for each_image in file_num:
            image = cv_imread(each_image)
            each_image = "test11\\test\\J01_2018.06.16 10_17_26.jpg"
            print(each_image)
            for i in range(divwidth):
                for j in range(divheight):
                    if j == 4 and i == 4:
                        x_startpoint = 1470
                        x_endpoint = 1920
                        y_startpoint = 1960
                        y_endpoint = 2560
                    elif j == 4:
                        x_startpoint = 1470
                        x_endpoint = 1920
                        y_startpoint = i * 600
                        y_endpoint = (i + 1) * 600
                    elif i == 4:
                        x_startpoint = j * 450
                        x_endpoint = (j + 1) * 450
                        y_startpoint = 1960
                        y_endpoint = 2560
                    else:
                        x_startpoint = j * 450
                        x_endpoint = (j + 1) * 450
                        y_startpoint = i * 600
                        y_endpoint = (i + 1) * 600
                    img = image[x_startpoint:x_endpoint, y_startpoint:y_endpoint]
                    img1 = cv2.transpose(img, 0)
                    imgarr = img_to_array(img1)
                    image_arr = np.array(imgarr, dtype="float32") / 255.0
                    image_arr = image_arr.reshape((1, 600, 450, 3))
                    result = model.predict(image_arr)
                    print(result)

    def predict(self):
        model = load_model("resnet50_model_weights.h5")
        image_path = "test11\\test"
        csvFile = open("submit.csv", "w", newline='')
        writer = csv.writer(csvFile)
        list_head = ['filename', 'probability']
        writer.writerow(list_head)
        file_num = list(paths.list_images(image_path))
        for each_image in file_num:
            image = cv_imread(each_image)
            img1 = cv2.transpose(image, 0)
            #image1 = cv2.resize(img1, (1000, 750))
            imgarr = img_to_array(img1)
            image_arr = np.array(imgarr, dtype="float") / 255.0
            # image_arr = np.array(imgarr)
            image_arr = image_arr.reshape((1, 1000, 750, 3))
            result = model.predict(image_arr)
            probility = round(result[0][1], 6)
            each_file = each_image.replace('\ ', ' ')
            each_file = os.path.basename(each_file)
            add_info = [each_file, probility]
            writer.writerow(add_info)
        csvFile.close()


if __name__ == '__main__':
    img_row = 1000
    img_col = 750
    batch_size = 2

    transfer = PowerTransferMode()

    # 得到数据
    train_generator = transfer.DataGen('twoclasscli\\train', img_row, img_col, batch_size, True)
    validation_generator = transfer.DataGen('twoclasscli\\test', img_row, img_col, batch_size, False)

    # VGG19
    # model = transfer.VGG19_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # history_ft = transfer.train_model(model, 10, train_generator, 600, validation_generator, 60, 'vgg19_model_weights.h5', is_load_model=False)

    # ResNet50
    # model = transfer.ResNet50_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # history_ft = transfer.train_model(model, 10, train_generator, 600, validation_generator, 60,
    #                                   'resnet50_model_weights.h5', is_load_model=False)

    # xception模型
    model = transfer.Xception_model(nb_classes=2, img_rows=img_col, img_cols=img_row, is_plot_model=False)
    history_ft = transfer.train_model(model, 50, train_generator, 50, validation_generator, 2,
                                      'resnet50_model_weights.h5', is_load_model=False)

    # InceptionV3
    # model = transfer.InceptionV3_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=True)
    # history_ft = transfer.train_model(model, 10, train_generator, 600, validation_generator, 60, 'inception_v3_model_weights.h5', is_load_model=False)

    # 训练的acc_loss图
    transfer.plot_training(history_ft)
    # transfer.predict()
