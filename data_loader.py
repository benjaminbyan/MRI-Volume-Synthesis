from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns
import numpy as np 
import pandas as pd  

from IPython import display
from IPython.display import clear_output

from data_augmentation import load, augmentation, normalize

BUFFER_SIZE=400
BATCH_SIZE=1

height = 256
width = 256

def load_image_train(image_file):
    input_image, real_image = normalize(augmentation(load(image_file)))
    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = normalize(resize(load(image_file)[0],load(image_file)[1],height,width))
    return input_image, real_image

def load_image_train_T1T2(image_file):
   input_image, real_image = normalize(augmentation(load(image_file)))
   return input_image, real_image

def load_image_train_T2T1(image_file):
   input_image, real_image = normalize(augmentation(load(image_file)))
   return input_image, real_image

def load_image_test_T1T2(image_file):
    input_image, real_image = normalize(resize(load(image_file)[0],load(image_file)[1],height,width))
    return input_image, real_image


def load_image_test_T2T1(image_file):
    input_image, real_image = normalize(resize(load(image_file)[0],load(image_file)[1],height,width))
    return input_image, real_image

#adjust the file path based on the specific mapping/generation task
Path1 = "./Flair/"

#### training pipeline ########

train_dataset_T1T2 = tf.data.Dataset.list_files(Path1+"Training/*.jpg")
train_dataset_T2T1 = tf.data.Dataset.list_files(Path1+"Training/*.jpg")
train_dataset_T1T2 = train_dataset_T1T2.map(load_image_train_T1T2,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_T2T1 = train_dataset_T2T1.map(load_image_train_T2T1,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_T1T2 = train_dataset_T1T2.shuffle(BUFFER_SIZE)
train_dataset_T1T2 = train_dataset_T1T2.batch(BATCH_SIZE)
train_dataset_T2T1 = train_dataset_T2T1.shuffle(BUFFER_SIZE)
train_dataset_T2T1 = train_dataset_T2T1.batch(BATCH_SIZE)

##### testing pipeline #####

test_dataset_T1T2 = tf.data.Dataset.list_files(Path1+'Testing/*.jpg')
test_dataset_T2T1 = tf.data.Dataset.list_files(Path1+'Testing/*.jpg')
test_dataset_T1T2 = test_dataset_T1T2.map(load_image_test_T1T2)
test_dataset_T2T1 = test_dataset_T2T1.map(load_image_test_T2T1)
test_dataset_T1T2 = test_dataset_T1T2.batch(BATCH_SIZE)
test_dataset_T2T1 = test_dataset_T2T1.batch(BATCH_SIZE)
