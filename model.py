from keras.models import Sequential, Model,load_model
from keras.layers import Convolution2D,Cropping2D,Dropout,MaxPooling2D,Lambda
from keras.layers.core import Dense,Activation,Flatten
import keras.preprocessing.image
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm
import os
import cv2

## Dear Constants and hyper parameters #####
DATA_DIR = "data/"
DRIVING_LOG = DATA_DIR + "driving_log.csv"
LEARNING_RATE = 0.0001
EPOCHS = 5
MODEL_SAVE = "model.h5" # Weights needed by drive.py
ANGLE_THRESHOLD = 0.25

print("Loading Driving log ....")
df = pd.read_csv(DRIVING_LOG)
# Use Pareto principle
print("Splitting Data ....")
train_data,validation_data = train_test_split(df,train_size=0.8)
###################################

# Thanks
def coin_flip():
    return np.random.randint(0,1)

def augment_brightness_camera_images(image):
    # Augment brightness of some images
    if coin_flip():
        augmented_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        augmented_image[:,:,2] = augmented_image[:,:,2]*random_bright
        augmented_image = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return augmented_image
    return image

def random_flip(image,steering):

    if coin_flip():
        image,steering=cv2.flip(image,1),-steering
    return image,steering

def preprocess_images(image):
    # Change brightness
    image = augment_brightness_camera_images(image)

    return image


def create_model():
    print("Creating Model")
    # This model is inspired from comma.ai's research model
    # Create a Sequential Model
    model = Sequential()
    # Add Cropping2D Layer to crop out 50 row pixels from top and 20 from bottom
    # Cropping2D leverages GPU's parallelism features to speed up training

    ###### Layer 1 #######################################################
   # model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Output shape is (None,90,320,3)
    #######################################################################

    ###### Layer 2 ########################################################
    # 2D Convolutional
    model.add(Convolution2D(32, 5, 5,subsample=(2, 2),border_mode="same"))
    # Followed by a ReLu Activation
    model.add(Activation('relu'))
    # Droput 45 %
    model.add(Dropout(0.45))
    # Max Pool Samples
    model.add(MaxPooling2D((2,2),border_mode='valid'))
    #######################################################################

    ####### Layer 3 #######################################################
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(MaxPooling2D((2,2),border_mode='valid'))
    #########################################################################

    ####### Layer 4 #################################################
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode="same"))
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

def chose_camera(df_row,training=False):
    tri_coin = np.random.randint(0,3)

    if tri_coin == 0 :
        img = plt.imread(DATA_DIR + df_row['left'].strip())
        steering_angle = df_row['steering'] + ANGLE_THRESHOLD

    elif tri_coin == 1:
        img = plt.imread(DATA_DIR + df_row['center'].strip())
        steering_angle = df_row['steering']

    else:
        img = plt.imread(DATA_DIR + df_row['right'].strip())
        steering_angle = df_row['steering'] - ANGLE_THRESHOLD

    if training:
        return random_flip(img,steering_angle)
    else:
        return (img,df_row['steering']) # For validation return unaltered data

def get_batch_data(batch_size,training):

    batch_data = []
    data_source = df
    # Generate random indexes from 0 to data frame size equal to batch sizes
    #if training:
    #    data_source = train_data
    #else:
    #    data_source = validation_data
    rand_index = np.random.randint(0,len(data_source),batch_size)
    for index in rand_index:
        # Randomly load one of the three images for a steering angle
        #print(data_source.ix[index])
        batch_data.append(chose_camera(data_source.ix[index],training))
    return batch_data



def generate_batch(batch_size,training=False):

     # http://stackoverflow.com/a/569063
    batch_images = [] # placeholder for images
    batch_steering = [] # placeholder for steering
    batch_data = get_batch_data(batch_size,training)
    while True:
        for img,angle in batch_data:
            batch_images.append(img)
            batch_steering.append(angle)
        yield np.array(batch_images), np.array(batch_steering)

def save_model(model):
    print("Saving Model ....")
    model.save(MODEL_SAVE)

def train():
    print("Training ....")
    # Create the model
    model = create_model()
    print("Model created ....")
    training_generator = generate_batch(32,training=True)
    validation_generator = generate_batch(32)

    history = model.fit_generator(training_generator,nb_epoch=EPOCHS,validation_data=validation_generator,verbose=1,samples_per_epoch=64,nb_val_samples=64)
    save_model(model)

train()
