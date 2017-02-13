from keras.models import Sequential, Model
from keras.layers import Convolution2D,Cropping2D,Dropout,MaxPooling2D,Lambda
from keras.core import Dense,Activation,Flatten
import keras.preprocessing.image
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm
import os
import cv2

## Dear Constants and hyper parameters #####
DATA_DIR = "data"
DRIVING_LOG = data_dir + "/driving_log.csv"
LEARNING_RATE = 0.0001
EPOCHS = 10
###################################

def augment_brightness_camera_images(image):
    augmented_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    augmented_image[:,:,2] = augmented_image[:,:,2]*random_bright
    augmented_image = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return augmented_image


def create_model():
    # This model is inspired from comma.ai's research model
    # Create a Sequential Model
    model = Sequential()
    # Add Cropping2D Layer to crop out 50 row pixels from top and 20 from bottom
    # Cropping2D leverages GPU's parallelism features to speed up training

    ###### Layer 1 #######################################################
    model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Output shape is (None,90,320,3)
    #######################################################################

    ###### Layer 2 ########################################################
    # 2D Convolutional
    model.add(Convolution2D(32, 5, 5,subsample=(2, 2)),border_mode="same")
    # Followed by a ReLu Activation
    model.add(Activation('relu'))
    # Droput 45 %
    model.add(Dropout(0.45))
    # Max Pool Samples
    model.add(MaxPooling2D((2,2),border_mode='valid'))
    #######################################################################

    ####### Layer 3 #######################################################
    model.add(Convolution2D(16,3,3,subsample=(1,1)),border_mode="same")
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(MaxPooling2D((2,2),border_mode='valid'))
    #########################################################################

    ####### Layer 4 #################################################
    model.add(Convolution2D(16,3,3,subsample=(1,1)),border_mode="same")
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ##################################################################

    ####### Layer 5 ###################################################
    model.add(Flatten())

    # Add Dense Layer similar to Nvidia's End to End Learning Model
    model.add(Dense(1116))
    # Reduce dropout to 35 % for connected layers
    model.add(Dropout(.35))
    # Activate
    model.add(Activation('relu'))
    ####################################################################

    ####### Layer 6 #######################################
    model.add(Dense(100))
    # Gradually reduce dropout with density
    model.add(Dropout(.3))
    # Activate
    model.add(Activation('relu'))
    #################################################

    ###### Layer 7 #####################################
    model.add(Dense(50))
    model.add(Dropout(.1))
    model.add(Dense(10))
    ################################################

    ####### Layer 8 ################################
    # Since we need prediction of steering angle, only single neuron/output for that
    model.add(Dense(1))
    #################################################

    ######## Specifying Algorithm and loss function ###########
    # Adam optimizer and Mean Squared Error (MSE) as loss function
    model.compile(optimizer="adam",loss="mse")
    ##########################################################

    return model


def train():
    # Read the CSV File
    df = pd.read_csv(driving_log)
