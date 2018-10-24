from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
from imutils import paths
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import sys
sys.path.append("..")
import cv2

class_number = 2

def cv_imread(file_path):
    path = file_path.replace('\ ',' ')
    cv_image = cv2.imdecode(np.fromfile(path , dtype = np.uint8) , -1)
    return cv_image

def split_file(path):
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagepaths)
    random.seed(72)
    random.shuffle(imagepaths)
    random.seed(62)
    random.shuffle(imagepaths)
    random.seed(83)
    random.shuffle(imagepaths)
    file_number = len(imagepaths)
    rate = int(file_number * 0.70)
    train_data = imagepaths[0:rate]
    verification_data = imagepaths[rate:file_number]
    return train_data,verification_data

def batch_test_data(train_data):
    data = []
    labels = []
    for imagepath in train_data:
        image = cv_imread(imagepath)
        image = cv2.resize(image,(1000,750))
        image = img_to_array(image)
        image = image/255
        data.append(image)
        label = imagepath.split(os.path.sep)[1]
        if label != '正常':
             labels.append(1)
        else:
             labels.append(0)
    data = np.array(data)
    print(data.shape)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=class_number)
    return data, labels

def batch_train_data(train_data):
    data = []
    labels = []
    for imagepath in train_data:
        image = cv_imread(imagepath)
        image = cv2.resize(image,(1000,750))
        image = img_to_array(image)
        image = image / 255
        data.append(image)

        label = imagepath.split(os.path.sep)[1]
        if label != '正常':
             labels.append(1)
        else:
             labels.append(0)
    data = np.array(data)
    print(data.shape)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=class_number)
    return data,labels


def batch_verification_file(verification_data):
    while 1:
        data = []
        labels = []
        list_label = []
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


def spltwo_data(path):
    print("[info]:loading image")
    data = []
    labels = []
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagepaths)
    for imagepath in imagepaths:
        image = cv_imread(imagepath)
        image = cv2.resize(image,(1000,750))
        image = img_to_array(image)
        image = image / 255
        data.append(image)

        label = imagepath.split(os.path.sep)[1]
        if label != '正常':
             labels.append(1)
        else:
             labels.append(0)
    data = np.array(data)
    labels = np.array(labels)
    print("data:",data.shape)
    print("labels:",labels.shape)
    (trainx ,testx,trainy,testy) = train_test_split(data,labels,test_size=0.3,random_state=0)
    trainy = to_categorical(trainy,class_number)
    testy = to_categorical(testy,class_number)
    return trainx,trainy,testx,testy



