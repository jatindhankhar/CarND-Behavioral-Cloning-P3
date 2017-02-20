import helper
import tensorflow as tf
import pandas as pd
from keras.models import Sequential, Model,load_model
from keras.layers import Convolution2D,Cropping2D,Dropout,MaxPooling2D,Lambda
from keras.layers.core import Dense,Activation,Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt




def save_model(model,name="model.h5"):
    print("Saving Model ...")
    model.save(name)


def create_model():
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    adam = Adam(lr=0.0001) # Use adam optimizer with a learning rate of 0.001
    model.compile(optimizer=adam, loss='mean_squared_error')

    return model

def train():
    print("Loading data ...")
    df = helper.read_data()
    # Follow Paleto's principle for splitting training and validation data
    print("Splitting data ...")
    df_train, df_valid = train_test_split(df, test_size=0.2)

    print("Creating model ...")
    model = create_model()
    # Train the model
    #samples_per_epoch = (20000//128)*128
    print("Training model ...")
    tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history_object = model.fit_generator(
    helper.data_generator(df_train,training=True),
    samples_per_epoch=len(df_train),
    nb_epoch=5,
    validation_data=helper.data_generator(df_valid,training=False),
    nb_val_samples=len(df_valid), callbacks=[tbCallback])

    save_model(model)
    print("Done ... :)")
    print("visualizing ...")
    print(history_object.history.keys())

### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == "__main__" :
    train()
