'''''''''''''''''''''''''''''''''''''''''''''''''''''''
#自定义学习率
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
'''''''''''''''''''''''''''''''''''''''''''''''''''''''
from keras.applications import Xception
from keras.optimizers import RMSprop
from keras.regularizers import l2,l1,l1_l2
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
basic_model = Xception(include_top=False, weights='imagenet', pooling='avg')
#冻结层
for layer in basic_model.layers:
    layer.trainable = False    #冻结层

model = Model(inputs=input_tensor, outputs=x)
model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

for layer in model.layers:
    layer.W_regularizer = l2(1e-2) #j加入L2正则化惩罚想
    layer.b_regularizer = l1(1e-2)
    layer.activity_regularizer = l1_l2(1e-2)
    layer.trainable = True  #解冻层

model.compile(optimizer=RMSprop(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='./checkpoint/weights_xception.h5', verbose=1,
                               save_best_only=True)
lr = LearningRateScheduler(lr_schedule)
model.fit_generator(train_generator,
                    steps_per_epoch=400,
                    epochs=150,
                    validation_data=val_generator,
                    callbacks=[checkpointer, tensorboard, lr],
                    initial_epoch=40,
                    workers=4,
                    verbose=0)



