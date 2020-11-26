#the code specifically used for T1 to FLAIR translation and mapping

from skimage import data
from skimage.color import rgb2gray

import sklearn
from sklearn import metrics

import numpy as np
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import math

from PIL import Image

Path_Images = "./Generator_Images/"

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

from IPython.display import clear_output

from generator import Generator 
from classifier import Discriminator
from model import generator_loss, discriminator_loss, calc_cycle_loss

from data_loader import load_image_train, load_image_test
from data_loader import load_image_train_T1T2, load_image_train_T2T1
from data_loader import load_image_tesr_T1T2, load_image_test_T2T1

from data_augmentation import load, augmentation, normalize

BUFFER_SIZE=400
BATCH_SIZE=1

height = 256
width = 256

#bringing in the data

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

#engineering all four neural networks including the forward and inverse generators,
#as well as the classifiers/discriminators for T1 and T2 images (was revised for other mappings using other data directories)
generator_T1T2 = Generator()
generator_T2T1 = Generator()

discriminator_T1T2 = Discriminator()
discriminator_T2T1 = Discriminator()

#building the Adam optimizers for each of the neural networks

generator_T1T2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_T2T1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_T1T2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_T2T1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#using checkpoints
checkpoint_path = "checkpoints/Train"

ckpt = tf.train.Checkpoint(generator_T1T2=generator_T1T2,
                           generator_T2T1=generator_T2T1,
                           discriminator_T1T2 = discriminator_T1T2,
                           discriminator_T2T1 = discriminator_T2T1,
                           generator_T1T2_optimizer=generator_T1T2_optimizer,
                           generator_T2T1_optimizer=generator_T2T1_optimizer,
                           discriminator_T1T2_optimizer=discriminator_T1T2_optimizer,
                           discriminator_T2T1_optimizer=discriminator_T2T1_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print("Checkpoint Restored")


EPOCHS = 60

LAMBDA2 = 10



ZEROES = 5

def image_number(n):
    number = str(n)

    while len(number) < ZEROES:
        number = "0" + number
    
    return number

def generate_images_modern_T1T2(model, test_input, tar, epoch, i):
    intel = i
    epoch_1 = epoch
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    prediction_gray = rgb2gray(prediction_map)
    tar_gray = rgb2gray(tar_map)

    error = np.abs(prediction_gray[:,:]-tar_gray[:,:])

    display_list = [test_input[0], tar[0], prediction[0], error]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', "Error Image"]
    
    #### mean squared error ####
    loss1 = sklearn.metrics.mean_squared_error(tar_gray, prediction_gray)
    loss1 = round(loss1, 3)

    ### structural similarity index #####
    loss2 = ssim(tar_gray, prediction_gray)
    loss2 = round(loss2, 3)
    
    ##### peak signal to noise ratio (PSNR) ########
    
    loss3 = 20*math.log(4095/loss1, 10)
    loss3 = round(loss3, 3)
    
    ##### mean absolute error #######
    loss4 = sklearn.metrics.mean_absolute_error(tar_gray, prediction_gray)
    loss4 = round(loss4, 3)
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i], fontsize = 16)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.suptitle("Epoch: "+str(epoch_1)+"         MSE: "+str(loss1)+"      SSI "+str(loss2)+"    PSNR: "+str(loss3)+ "     MAE: "+str(loss4), x=0.5, y = 0.95, verticalalignment = "center", fontsize=22)
    #plt.show()
    
    plt.savefig(os.path.join(Path_Images, "T1T2"+image_number(epoch_1)+"_"+image_number(intel)+".jpg"))
    
def generate_images_modern_T2T1(model, test_input, tar, epoch, i):
    intel = i
    epoch_1 = epoch
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    prediction_gray = rgb2gray(prediction_map)
    tar_gray = rgb2gray(tar_map)

    error = np.abs(prediction_gray[:,:]-tar_gray[:,:])

    display_list = [test_input[0], tar[0], prediction[0], error]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', "Error Image"]
    
    #### mean squared error ####
    loss1 = sklearn.metrics.mean_squared_error(tar_gray, prediction_gray)
    loss1 = round(loss1, 3)

    ### structural similarity index #####
    loss2 = ssim(tar_gray, prediction_gray)
    loss2 = round(loss2, 3)
    
    ##### peak signal to noise ratio (PSNR) ########
    
    loss3 = 20*math.log(4095/loss1, 10)
    loss3 = round(loss3, 3)
    
    ##### mean absolute error #######
    loss4 = sklearn.metrics.mean_absolute_error(tar_gray, prediction_gray)
    loss4 = round(loss4, 3)
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i], fontsize = 16)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.suptitle("Epoch: "+str(epoch_1)+"         MSE: "+str(loss1)+"      SSI "+str(loss2)+"    PSNR: "+str(loss3)+ "     MAE: "+str(loss4), x=0.5, y = 0.95, verticalalignment = "center", fontsize=22)
    #plt.show()
    
    plt.savefig(os.path.join(Path_Images, "T2T1"+image_number(epoch_1)+"_"+image_number(intel)+".jpg"))

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    #visualizing the output 
    for i in range(4):
      for real_T1, real_T2 in test_ds.take(1):
        generate_images_modern_T1T2(generator_T1T2, real_T1, real_T2, epoch, i)
        generate_images_modern_T2T1(generator_T2T1, real_T2, real_T1, epoch, i)
        
    print("Epoch: ", epoch)

    # Training the neural networks
    for n, (real_T1, real_T2) in train_ds.enumerate():
      print('.', end='') #visualizing the progress
      if (n+1) % 100 == 0:
        print()
      train_step(real_T1, real_T2, epoch)
    print()
    
    #saving out the model every 3 epochs of training
    if (epoch + 1) % 3 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                     ckpt_save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                  time.time()-start))

#fitting the neural network to the data
fit(train_dataset_T1T2, EPOCHS, test_dataset_T1T2)
