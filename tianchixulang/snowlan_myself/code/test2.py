import numpy as np
import keras
from keras.utils import np_utils
from PIL import Image
import random
from imutils import paths
import cv2
from keras.preprocessing.image import img_to_array
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
CLASS_NUM = 2

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

def batch_train_data(train_data,batch_size):
    while 1:
        cnt = 0
        data = []
        labels = []
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = img_to_array(image)
            image_arr =  np.array(image, dtype="float32") / 255.0
            data.append(image_arr)
            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                labels = np.array(labels)
                num = 0
                for each_label in labels:
                    if each_label != '正常':
                        labels[num] = '疵点'
                    num += 1
                labels_inter = LabelEncoder().fit_transform(labels)
                labels = to_categorical(labels_inter, num_classes=CLASS_NUM)
                yield (data, labels)
                data = []
                labels = []

def batch_verification_file(verification_data):
    data = []
    labels = []
    for imagepath in verification_data:
        image = cv_imread(imagepath)
        image = img_to_array(image)
        image_arr =  np.array(image, dtype="float32") / 255.0
        data.append(image_arr)

        # extract the class label
        label = imagepath.split(os.path.sep)[1]
        labels.append(label)

    #data = np.array(data, dtype="float") / 255.0 #X_train = X_train.astype('float32')/255
    labels = np.array(labels)
    num = 0
    for each_label in labels:
        if each_label != '正常':
            labels[num] = '疵点'
        num += 1
    labels_inter = LabelEncoder().fit_transform(labels)
    labels = to_categorical(labels_inter, num_classes=CLASS_NUM)
    return data,labels

train_file,verification_file = split_file("xuelang\\")

f = batch_train_data(train_file,32)
f.__next__
print("tarin")
ytrain ,ytest = batch_verification_file(verification_file)
print(len(ytest))
print("it is over")
