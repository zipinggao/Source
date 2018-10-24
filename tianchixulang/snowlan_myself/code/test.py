from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import os
import sys
sys.path.append("..")
import cv2

def cv_imread(file_path):
    path = file_path.replace('\ ',' ')
    cv_image = cv2.imdecode(np.fromfile(path , dtype = np.uint8) , -1)
    return cv_image

def load_data(path):
    print("[info]:loading image")
    data = []
    labels = []
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagepaths)
    for imagepath in imagepaths:
        image =cv_imread(imagepath)
        #image = cv2.resize(image , (1920 ,2560))

        image = img_to_array(image)
        data.append(image)

        #extract the class label
        label = imagepath.split(os.path.sep)[1]
        labels.append(label)

    #liner map to [0 , 1]
    data = np.array(data , dtype = "float")/255
    labels = np.array(labels)
    #convert the labels from integers to vector
    #set_labels = list(set(labels))
    #index = range(1,len(set(labels))+1)
    #traing data 0.75 ,test data 0.25
    '''
    初赛部分
    '''
    num = 0
    for each_label in labels:
        if each_label != '正常':
            labels[num] = '疵点'
        num +=1
    labels_inter = LabelEncoder().fit_transform(labels)
    (trainx ,testx,trainy,testy) = train_test_split(data , labels_inter,test_size=0.25,random_state=42)
    trainy = to_categorical(trainy , num_classes = len(set(trainy)))
    testy = to_categorical(trainy , num_classes = len(set(testy)) )
    return trainx, trainy,testx,testy

def load_data1(path):
    print("[info]:loading image")
    data = []
    labels = []
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagepaths)
    print(len(imagepaths))
    for imagepath in imagepaths:
        image =cv_imread(imagepath)
        #image = cv2.resize(image , (1920 ,2560))

        image = img_to_array(image)
        #image = np.array(image, dtype="float") / 255.0
        data.append(image)

        #extract the class label
        label = imagepath.split(os.path.sep)[1]
        labels.append(label)

    #liner map to [0 , 1]
    data = np.array(data , dtype = "float") /255.0
    labels = np.array(labels)

    #convert the labels from integers to vector
    #set_labels = list(set(labels))
    #index = range(1,len(set(labels))+1)
    #traing data 0.75 ,test data 0.25
    labels_inter = LabelEncoder().fit_transform(labels)
    (trainx ,testx,trainy,testy) = train_test_split(data , labels_inter,test_size=0.25,random_state=42)
    trainy = to_categorical(trainy , num_classes = len(set(trainy)))
    testy = to_categorical(trainy , num_classes = len(set(testy)) )
    return trainx, trainy,testx,testy


load_data("xuelang\\")