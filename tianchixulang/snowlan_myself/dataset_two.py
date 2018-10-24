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

class_number = 2;

def cv_imread(file_path):
    path = file_path.replace('\ ',' ')
    cv_image = cv2.imdecode(np.fromfile(path , dtype = np.uint8) , -1)
    return cv_image

def split_file(path):
    imagepaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagepaths)
    random.seed(82)
    random.shuffle(imagepaths)
    random.seed(53)
    random.shuffle(imagepaths)
    random.seed(73)
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
        list_label = []
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = cv2.resize(image,(600,450))
            image = img_to_array(image)
            image = image/255
            data.append(image)

            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                data = np.array(data)
                labels = np.array(labels)
                for each_label in labels:
                    if each_label != '正常':
                        list_label.append(1)
                    else:
                        list_label.append(0)

                #labels_inter = LabelEncoder().fit_transform(labels)

                labels = to_categorical(list_label, num_classes=class_number)
                yield (data, labels)
                data = []
                labels = []
                list_label= []


def batch_train_data(train_data,batch_size):
    while 1:
        cnt = 0
        data = []
        labels = []
        list_label = []
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = cv2.resize(image,(600,450))
            image = img_to_array(image)
            image = image / 255
            data.append(image)

            label = imagepath.split(os.path.sep)[1]
            labels.append(label)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                data = np.array(data)
                labels = np.array(labels)
                for each_label in labels:
                    if each_label != '正常':
                        list_label.append(1)
                    else:
                        list_label.append(0)

                #labels_inter = LabelEncoder().fit_transform(labels)

                labels = to_categorical(list_label, num_classes=class_number)
                yield (data, labels)
                data = []
                labels = []
                list_label= []

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



def test_data(train_data,batch_size):
    while 1:
        cnt = 0
        data = []
        labels = []
        list_label = []
        flag = 0
        for imagepath in train_data:
            image = cv_imread(imagepath)
            image = img_to_array(image)
            if cnt == 0:
                image1 = np.array(image, dtype="float32") / 255.0
            else:
                image_test = np.array(image, dtype="float32") / 255.0
                flag = 1
            if flag == 1:
                image_test2 = np.append(image1 , image_test)
                flag +=1
            elif flag > 1:
                image_test2 = np.append(image_test2 , image_test)
                print(image_test2.shape)
            #image = cv2.resize(image,(400,300))

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
                        #labels[num] = '疵点'
                        # labels[num] = 1
                        # labels[num]
                        list_label.append(1)
                    else:
                        #labels[num] = 0
                        list_label.append(0)

                    num += 1
                #labels_inter = LabelEncoder().fit_transform(labels)

                labels = to_categorical(list_label, num_classes=class_number)

                print("datatrain:",data.shape)
                # print("labeltrain:",labels.shape)
                # exit()
                yield (data, labels)
                data = []
                labels = []
                list_label= []
