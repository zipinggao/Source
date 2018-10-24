from dataset import cv_imread
import csv
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array ,ImageDataGenerator
import numpy as np
from imutils import paths
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config1)

image_height = 750
image_width = 1000
batch_size = 4
nb_testsamples = 1000
nb_number = 1
def predictcorss():
    model = load_model('xuelang_xception_tuned5.h5')
    test_datagen = ImageDataGenerator(
        rescale=1./255
        )
    for idx in range(nb_number):
        random_seed = np.random.random_integers(0,662)
        test_generator = test_datagen.flow_from_directory(
            'test11/',
            target_size = (image_height, image_width),
            batch_size = batch_size,
            class_mode = None,
            shuffle = False,
            seed = random_seed,
            classes = None
        )
        test_image_list = test_generator.filenames
        if idx == 0:
            predictions = np.array(model.predict_generator(test_generator))
        else:
            predictions +=np.array(model.predict_generator(test_generator))
    predictions /= nb_number
    csvFile = open("5-submit-cross.csv", "w", newline='')
    writer = csv.writer(csvFile)
    list_head = ['filename', 'probability']
    writer.writerow(list_head)
    for i in range(662):
        if predictions[0][i][0]> 0.5 and predictions[2][i][0]> 0.5:
            if predictions[0][i][0] > predictions[2][i][0]:
                probility = predictions[0][i][0]
            else:
                probility = predictions[2][i][0]
        elif predictions[0][i][0]< 0.5 and predictions[2][i][0]< 0.5:
            if predictions[0][i][0] > predictions[2][i][0]:
                probility =predictions[2][i][0]
            else:
                probility = predictions[0][i][0]
        else:
            probility =(predictions[0][i][0] +  predictions[2][i][0])/2
        probility = round(probility, 6)
        add_info = [test_image_list[i], probility]
        # add_info = [test_image_list[i],round(predictions[0][i][0],6), round(predictions[1][i][0],6), round(predictions[2][i][0],6), round(predictions[3][i][0],6)]
        writer.writerow(add_info)
    csvFile.close()

predictcorss()
