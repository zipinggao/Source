import os
import numpy as np
import matplotlib.pyplot as plt
from keras import Input
from keras.applications import Xception, InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, concatenate, maximum
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import tensorflow as tf
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
batch_size = 8
image_height = 960
image_width = 1280
image_deep = 3
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=90,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'twoclassoigial2/traindata',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary') #binary

validation_generator = test_datagen.flow_from_directory(
    'twoclassoigial2/valdata',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary') #binary


def triple_generator(generator):
    while True:
        x, y = generator.next()
        yield x, [y, y, y, y]

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                            cooldown=0, min_lr=0)

if os.path.exists('xulang_single_xception6.h5'):
    model = load_model('xulang_single_xception6.h5')
else:
    # create the base pre-trained model
    input_tensor = Input(shape=(image_height, image_width, image_deep))
    base_model1 = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    base_model1 = Model(inputs=[base_model1.input], outputs=[base_model1.get_layer('avg_pool').output], name='xception')

    base_model2 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    base_model2 = Model(inputs=[base_model2.input], outputs=[base_model2.get_layer('avg_pool').output],
                        name='inceptionv3')

    img1 = Input(shape=(image_height, image_width, image_deep), name='img_1')

    feature1 = base_model1(img1)
    feature2 = base_model2(img1)

    # let's add a fully-connected layer
    category_predict1 = Dense(1, activation='sigmoid', name='ctg_out_1')(
        Dropout(0.5)(
            feature1
        )
    )

    category_predict2 = Dense(1, activation='sigmoid', name='ctg_out_2')(
        Dropout(0.5)(
            feature2
        )
    )
    category_predict = Dense(1, activation='sigmoid', name='ctg_out')(
        concatenate([feature1, feature2])
    )
    max_category_predict = maximum([category_predict1, category_predict2])

    model = Model(inputs=[img1], outputs=[category_predict1, category_predict2, category_predict, max_category_predict])

    # model.save('dog_xception.h5')
    plot_model(model, to_file='single_mode4.png')

    for layer in base_model1.layers:
        layer.trainable = False

    for layer in base_model2.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='nadam',
                  loss={
                      'ctg_out_1': 'binary_crossentropy',  #categorical_crossentropy
                      'ctg_out_2': 'binary_crossentropy',
                      'ctg_out': 'binary_crossentropy',
                      'maximum_1': 'binary_crossentropy'
                  },
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    model.fit_generator(triple_generator(train_generator), #triple_generator
                        steps_per_epoch=2839 / batch_size + 1,
                        epochs=30,
                        validation_data=triple_generator(validation_generator),#triple_generator
                        validation_steps=205 / batch_size + 1,
                        callbacks=[auto_lr]
                       )

    model.save('xulang_single_xception6.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
cur_base_model = model.layers[1]
for layer in cur_base_model.layers[:105]:
    layer.trainable = False
for layer in cur_base_model.layers[105:]:
    layer.trainable = True

cur_base_model = model.layers[2]
for layer in cur_base_model.layers[:262]:
    layer.trainable = False
for layer in cur_base_model.layers[262:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss={
                      'ctg_out_1': 'binary_crossentropy',#categorical_crossentropy
                      'ctg_out_2': 'binary_crossentropy',
                      'ctg_out': 'binary_crossentropy',
                      'maximum_1': 'binary_crossentropy'
                  },
              metrics=['accuracy'])
# batch_size = batch_size * 3 / 4

print(batch_size)
train_generator = test_datagen.flow_from_directory(
    'twoclassoigial2/traindata',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary')  #binary
validation_generator = test_datagen.flow_from_directory(
    'twoclassoigial2/valdata',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary')  #binary  categorical

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
save_model = ModelCheckpoint('xception-66.h5', monitor="val_ctg_out_acc", verbose=1,
                                 save_best_only=True, save_weights_only=True,mode="max")
history = model.fit_generator(triple_generator(train_generator),#triple_generator
                    steps_per_epoch=2839 / batch_size + 1,
                    epochs=60,
                    validation_data=triple_generator(validation_generator), #triple_generator
                    validation_steps=205 / batch_size + 1,
                    callbacks = [auto_lr, save_model])
model.save('xuelang_xception_tuned6.h5')

acc = history.history['ctg_out_acc']
val_acc = history.history['val_ctg_out_acc']
loss = history.history['ctg_out_loss']
val_loss = history.history['val_ctg_out_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b-')
plt.plot(epochs, val_acc, 'r')
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'b-')
plt.plot(epochs, val_loss, 'r-')
plt.title('Training and validation loss')
plt.show()
