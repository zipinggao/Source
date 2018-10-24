import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'keras_retinanet\\bin\\snapshots', 'resnet50_pascal_24.h5')


# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.load_model(model_path, backbone_name='resnet50', convert=True)

print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names11 = {
    '边白印': 0,
    '厚薄段' : 1,
    '纬粗纱' : 2,
    '扎洞' : 3,
    '弓纱' : 4,
    '边扎洞' : 5,
    '扎纱' : 6,
    '粗纱' : 7,
    '织稀' : 8,
    '耳朵' : 9,
    '黄渍' : 10,
    '回边' : 11,
    '毛斑' : 12,
    '跳花' : 13,
    '污渍' : 14,
    '吊纬' : 15,
    '扎梳' : 16,
    '破洞' : 17,
    '结洞' : 18,
    '缺经' : 19,
    '修印': 20,
    '破边': 21,
    '楞断': 22,
    '明嵌线': 23,
    '厚段': 24,
    '毛洞': 25,
    '擦毛': 26,
    '边针眼': 27,
    '线印' : 28,
    '缺纬' : 29,
    '擦洞': 30,
    '吊弓': 31,
    '吊经': 32,
    '边缺纬': 33,
    '剪洞': 34,
    '毛粒': 35,
    '经跳花': 36,
    '夹码': 37,
    '油渍': 38,
    '边缺经': 39,
    '蒸呢印': 40,
    '紧纱': 41,
    '嵌结': 42,
    '织入': 43,
    '经粗纱': 44,
    '愣断': 45,
    '擦伤': 46
}

labels_to_names = {
    'aeroplane'   : 0,
    'bicycle'     : 1,
    'bird'        : 2,
    'boat'        : 3,
    'bottle'      : 4,
    'bus'         : 5,
    'car'         : 6,
    'cat'         : 7,
    'chair'       : 8,
    'cow'         : 9,
    'diningtable' : 10,
    'dog'         : 11,
    'horse'       : 12,
    'motorbike'   : 13,
    'person'      : 14,
    'pottedplant' : 15,
    'sheep'       : 16,
    'sofa'        : 17,
    'train'       : 18,
    'tvmonitor'   : 19
}


image = read_image_bgr('2008_000066.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# correct for image scale
boxes /= scale
# print("---------------------------")
# print(boxes)
# print("---------------------------")
# print(scores)
# print("---------------------------")
# print(labels)
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    print(score)
    print(label)
    if score < 0.1:
        break

    color = label_color(label)
    b = box.astype(int)
    draw_box(draw, b, color=color)
    print(label)
    label_name = list(labels_to_names.keys())[list(labels_to_names.values()).index(label)]
    caption = "{} {:.3f}".format(label_name, score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()