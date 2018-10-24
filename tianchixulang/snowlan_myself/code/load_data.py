#encoding:UTF-8
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
sys.path.append("..")
import cv2
from lenet2 import LeNet

image_width = 2560;
image_heiht = 1920;
class_number = 2;
learn_rate = 1e-3;
Epochs = 35;
BatchSize = 32;

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def cv_imread(file_path):
    path = file_path.replace('\ ',' ')
    cv_image = cv2.imdecode(np.fromfile(path , dtype = np.uint8) , -1)
    return cv_image


def split_file(path):
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(82)
    random.shuffle(imagepaths)
    random.seed(82)
    random.shuffle(imagepaths)
    file_number = len(imagepaths)
    rate = int(file_number * 0.75)
    train_data = imagepaths[0:rate]
    verification_data = imagepaths[rate:file_number]
    return train_data,verification_data

def batch_test_data(train_data,batch_size):
    while 1:
        cnt = 0
        data = []
        labels = []
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = img_to_array(image)

            image = img_to_array(image)
            data.append(image)

            # image_arr =  np.array(image, dtype="float32") / 255.0
            # data.append(image_arr)

            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                data = np.array(data, dtype="float32") / 255.0
                labels = np.array(labels)
                num = 0
                for each_label in labels:
                    if each_label != '正常':
                        labels[num] = '疵点'
                    num += 1
                labels_inter = LabelEncoder().fit_transform(labels)
                labels = to_categorical(labels_inter, num_classes=class_number)
                # print("datatrain:",data.shape)
                # print("labeltrain:",labels.shape)
                # exit()
                yield (data, labels)
                data = []
                labels = []

def batch_train_data(train_data,batch_size):
    while 1:
        cnt = 0
        data = []
        labels = []
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = img_to_array(image)

            image = img_to_array(image)
            data.append(image)

            # image_arr =  np.array(image, dtype="float32") / 255.0
            # data.append(image_arr)

            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                data = np.array(data, dtype="float32") / 255.0
                labels = np.array(labels)
                num = 0
                for each_label in labels:
                    if each_label != '正常':
                        labels[num] = '疵点'
                    num += 1
                labels_inter = LabelEncoder().fit_transform(labels)
                labels = to_categorical(labels_inter, num_classes=class_number)
                # print("datatrain:",data.shape)
                # print("labeltrain:",labels.shape)
                # exit()
                yield (data, labels)
                data = []
                labels = []

def batch_verification_file(verification_data):
    while 1:
        data = []
        labels = []
        for imagepath in verification_data:
            image = cv_imread(imagepath)
            image = img_to_array(image)

            image_arr =  np.array(image, dtype="float32") / 255.0
            data.append(image_arr[1])

            # extract the class label
            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
        data = np.array(data)
        labels = np.array(labels)
        num = 0
        for each_label in labels:
            if each_label != '正常':
                labels[num] = '疵点'
            num += 1
        labels_inter = LabelEncoder().fit_transform(labels)
        labels = to_categorical(labels_inter, num_classes=class_number)
        print("testdata:",data.shape)
        print("testlabel:",labels.shape)
        #return data,labels
        return data,labels

def train(train_file,verification_file):
    # initialize the model
    print("[info]:compiling model.....")
    model = LeNet.build(width = image_width , height= image_heiht, depth= 3 ,calsses = class_number)
    opt = Adam(lr=learn_rate , decay= learn_rate/Epochs)
    model.compile(loss="categorical_crossentropy" ,optimizer=opt ,metrics=["accuracy"])

    # train the network
    print("[info]:training model.....")
    H = model.fit_generator(batch_train_data(train_file ,batch_size=1),
                            validation_data=batch_test_data(verification_file ,batch_size=1),validation_steps = 2 ,epochs=Epochs,verbose=1,samples_per_epoch=1000)
    # save the model to disk
    print("[info] serializing network...")
    #model.save(args["model"])

    plt.style.use("ggplot")
    plt.figure()
    N = Epochs
    plt.plot(np.arange(0,N) ,H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #plt.savefig(args["plot"])

if __name__ == "__main__":
    file_path = "xuelang\\"
    train_file, verification_file = split_file("xuelang\\")
    #ytrain, ytest = batch_verification_file(verification_file)
    '''
    aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                             height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                             horizontal_flip=True,fill_mode="nearest")
                             '''
    train(train_file,verification_file)

