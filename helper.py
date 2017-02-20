import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import cv2
import random
import skimage.transform as sktransform
# Data directory contains
DATA_DIR = "data/"
CAMERAS = ['left','center','right']
STEERING_THRESHOLDS=[.25, 0., -.25]

def read_data():
    return pd.read_csv(DATA_DIR + "driving_log.csv")

def balance_data():
    return None

def select_camera():
    # Select one of the three cameras
    return np.random.randint(len(CAMERAS))

# Thanks https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.mz7nc5bsu
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# Thanks http://navoshta.com/end-to-end-deep-learning/
def random_shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image



def vertical_shift(image):
    top = int(random.uniform(.325, .425) * image.shape[0])
    bottom = int(random.uniform(.075, .175) * image.shape[0])
    image = image[top:-bottom, :]
    return image

def crop_and_resize(image,top_offset=.375, bottom_offset=.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

def mutate_and_augment(image):
    #image = augment_brightness_camera_images(image)
    image = random_shadow(image)
    #image = vertical_shift(image)
    return image

def preprocess_image(image,training=True):
    # Augment and mutate data for training only
    v_delta = 0
    if training:
       image = mutate_and_augment(image)
       v_delta = 0.05
    return crop_and_resize(image,top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta))


def data_generator(data,batch_size=128,training=True):
    while True:
        random_ix = np.random.permutation(data.index.values)
        for batches in range(0,len(random_ix),batch_size):
            batch_indices = random_ix[batches:(batches + batch_size)]
            images = []
            steering_angles = []
        # Preprocess images for batches
            for index in batch_indices:
                row = data.ix[index]
                if training:
                    selection = select_camera()
                    image = mpimg.imread(DATA_DIR + row[CAMERAS[selection]].strip())
                    steering = row.steering + STEERING_THRESHOLDS[selection]
                else:
                    image = mpimg.imread(DATA_DIR + row.center.strip())
                    steering = row.steering
                image = preprocess_image(image,training)
                images.append(image)
                steering_angles.append(steering)


                # Flip Half of data
            images = np.asarray(images)
            steering_angles = np.asarray(steering_angles)
            flip_indices = random.sample(range(images.shape[0]), int(images.shape[0] / 2))
            images[flip_indices] = images[flip_indices, :, ::-1, :]
            steering_angles[flip_indices] = -steering_angles[flip_indices]
            yield(images,steering_angles)
