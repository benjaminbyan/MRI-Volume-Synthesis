from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
from IPython.display import clear_output
AUTOTUNE = tf.data.experimental.AUTOTUNE
!pip install -U tensorboard

BUFFER_SIZE = 400
BATCH_SIZE = 1

#functions turn the images into 256 x 256 image arrays through resizing and data augmentation performing through 
#randomly cropping the image after padding it 286 x 286 and then flipping the image 
#across the vertical axis randomly

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input = image[:, :w, :]
    output = image[:, w:, :]

    input = input.cast(T1, tf.float32)
    output = output.cast(T2, tf.float32)

    return input, output

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
    stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    input_image = (input_image/127.5) - 1
    real_image = (real_image/127.5) - 1
    return input_image, real_image

def augmentation(input_image, real_image):

    heigth = 286
    width = 286

    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    input_image = tf.image.random_crop(tf.stack([input_image, real_image], axis=0), size=[2, IMG_HEIGHT, IMG_WIDTH, 3])[0]

    real_image = tf.image.random_crop(tf.stack([input_image, real_image], axis=0), size=[2, IMG_HEIGHT, IMG_WIDTH, 3])[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


