from dataset import cv_imread
import csv
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array ,ImageDataGenerator
import numpy as np
from imutils import paths
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config1)

File_path = 'test11\\test'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def predict():
    model = load_model('xuelang_xception_tuned5.h5')
    csvFile = open("submit_4.csv", "w", newline='')
    writer = csv.writer(csvFile)
    list_head = ['filename', 'probability']
    writer.writerow(list_head)
    file_num = list(paths.list_images(File_path))
    for each_file in file_num:
        image = cv_imread(each_file)
        image = cv2.resize(image, (1000, 750))
        image = img_to_array(image)
        image_arr = np.array(image, dtype="float32") / 255.0
        # image_arr = image_arr.reshape((1, 1920, 2560, 3))
        image_arr = image_arr.reshape((1, 750, 1000, 3))
        result = model.predict(image_arr)
        # probility = round(result[0][0], 6) #train4
        # probility = round(result[0][1], 6)

        each_file = each_file.replace('\ ', ' ')
        each_file = os.path.basename(each_file)

        # if result[0][0][0]> 0.5 and result[2][0][0]> 0.5:
        #     if result[0][0][0] > result[2][0][0]:
        #         probility = result[0][0][0]
        #     else:
        #         probility = result[2][0][0]
        # elif result[0][0][0]< 0.5 and result[2][0][0]< 0.5:
        #     if result[0][0][0] > result[2][0][0]:
        #         probility =result[2][0][0]
        #     else:
        #         probility = result[0][0][0]
        # else:
        #     probility =(result[0][0][0] +  result[2][0][0])/2

        add_info = [each_file, round(result[0][0][0],6),round(result[1][0][0],6),round(result[2][0][0],6),round(result[3][0][0],6)]
        # probility = round(probility ,6)
        # add_info = [each_file,probility]
        writer.writerow(add_info)
    csvFile.close()

predict()




