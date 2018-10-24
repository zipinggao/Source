from keras.applications  import Xception,InceptionV3
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout,AveragePooling2D
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.optimizers import SGD
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=configtf)
label_name =['complete','incomplete']

image_height = 210
image_width = 280
batchsize = 16
epochs = 30

def train():
    lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=3, min_lr=0, verbose=0)  # 设置学习率衰减
    # early_stopper = EarlyStopping(min_delta=0.001, patience=10,verbose=1)                                     #设置早停参数
    checkpoint = ModelCheckpoint("yue_alone_model.h5",
                                 monitor="val_acc", verbose=1,
                                 save_best_only=True, save_weights_only=True, mode="max")  # 保存训练过程中，在验证集上效果最好的模型
    train_data_aug = ImageDataGenerator(
        rotation_range=90,  # 图像旋转的角度
        width_shift_range=0.1,  # 左右平移参数
        height_shift_range=0.1,  # 上下平移参数
        zoom_range=0.1,  # 随机放大或者缩小
        horizontal_flip=True,  # 随机翻转  #// config.batch_size,
    )
    validation_data_aug = ImageDataGenerator(rescale=1. / 255)
    base_model = Xception(weights='imagenet', include_top=False,
                          input_shape=(image_height,image_width,3),
                         ) # classes=2,pooling='max'
    output = base_model.get_layer(index=-1).output
    output = AveragePooling2D((4,4),strides=(4,4),name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    # output = Dense(512, activation='relu')(output)
    # output = Dropout(0.4)(output)
    output = Dense(1,activation='sigmoid',name='predict')(output)
    model = Model(inputs=base_model.input, outputs=output)

    print("using data augmentation method")

    train_generator = train_data_aug.flow_from_directory(
        'twoclass/traindata',
        target_size=(image_height, image_width),
        batch_size = batchsize,
        shuffle=True,
        classes=label_name,
        class_mode='binary')  # binary
    validation_generator = validation_data_aug.flow_from_directory(
        'twoclass/valdata',
        target_size=(image_height, image_width),
        batch_size= batchsize,
        shuffle=True,
        classes=label_name,
        class_mode='binary')  # binary

    sgd = SGD(decay=0.0, momentum=0.9, lr=0.00001)
    model.compile(loss="binary_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])  # "adam"  categorical_crossentropy
    model.fit_generator(
        train_generator,
        steps_per_epoch=2013/batchsize + 1,
        validation_data=validation_generator,
        epochs=epochs,
        validation_steps= 105/batchsize + 1,
        callbacks=[lr_reducer, checkpoint]  # early_stopper
    )
    model.save("yueover_alone_model.h5")

if __name__ == "__main__":
    train()
