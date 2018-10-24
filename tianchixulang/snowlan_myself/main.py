from typical_network import typical_network
import dataset as data
import os
import tensorflow as tf
import keras
from keras.utils.training_utils import multi_gpu_model

gpu_num = 4
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


file_path = "xuelang11\\"
if __name__ == "__main__":
    # trainx, trainy, testx, testy = data.spltwo_data(file_path)
    model = typical_network()
    # model.Resnet_50()
    # model.vgg16()
    # model.googlenet()
    model.xceptionnet()
    # model.train(trainx, trainy, testx, testy)
    #model.evaluate_model(verification_file)
    # model.save()
    # model.predict()
    #model.best_predict()





