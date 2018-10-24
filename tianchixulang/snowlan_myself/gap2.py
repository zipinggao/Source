from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
import h5py
import tensorflow as tf
import os
import numpy as np
from sklearn.utils import shuffle
from IPython.display import SVG

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)


def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[1]
    height = image_size[0]
    print(width,height)
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("twoclass", image_size, shuffle=False,
                                              batch_size=8)

    test_generator = gen.flow_from_directory("test11", image_size, shuffle=False,
                                             batch_size=8, class_mode=None)
    train = model.predict_generator(train_generator ,train_generator.nb_sample)#, train_generator.nb_sample
    test = model.predict_generator(test_generator ,test_generator.nb_sample)#, test_generator.nb_sample
    with h5py.File("xuelang_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

# write_gap(ResNet50, (450, 600))
write_gap(Xception, (750, 1000), xception.preprocess_input)
# write_gap(InceptionV3, (450, 600), inception_v3.preprocess_input)
