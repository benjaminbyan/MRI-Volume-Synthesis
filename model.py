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

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#mean absolute error between the target image and synthesized image
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

#binary cross entropy loss of the real-synthetic classifier
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

LAMBDA2 = 10

#cycle-consistency loss of input image (D1 domain) and retranslated image (D1 -> D2 -> D1 mapping)
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

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

#function utilized for training the network with one epoch

#the entire training process is detailed in training.py

def train_step(real_T1, real_T2, epoch):

  with tf.GradientTape(persistent=True) as tape:
    synthetic_T2 = generator_T1T2(real_T1, training=True)
    cycled_T1 = generator_T2T1(synthetic_T2, training=True)
    
    synthetic_T1 = generator_T2T1(real_T2, training = True)
    cycled_T2 = generator_T1T2(synthetic_T1, training = True)
    
    input_image = real_T1
    target = real_T2
    
    disc_real_T1 = discriminator_T1T2([input_image, target], training = True)
    
    input_image = real_T2
    target = real_T1
    
    disc_real_T2 = discriminator_T2T1([input_image, target], training = True)
    
    disc_synthetic_T1 = discriminator_T1T2([real_T1, synthetic_T2], training = True)
    disc_synthetic_T2 = discriminator_T2T1([real_T2, synthetic_T1], training = True)

    generator_T1T2_loss, gen_gan_loss_T1T2, gen_l1_loss_T1T2 = generator_loss(disc_synthetic_T2, synthetic_T2, real_T2)
    generator_T2T1_loss, gen_gan_loss_T2T1, gen_l1_loss_T2T1 = generator_loss(disc_synthetic_T1, synthetic_T1, real_T1)
    
    #adds together cycle-consistency loss from T1 -> T2 -> T1 as well as T2 -> T1 -> T2
    total_cycle_loss = calc_cycle_loss(real_T1, cycled_T1) + calc_cycle_loss(real_T2, cycled_T2)
 
    #generator loss is MAE loss plus cycle-consistency loss
    total_generator_T1T2_loss = generator_T1T2_loss + total_cycle_loss
    total_generator_T2T1_loss = generator_T2T1_loss + total_cycle_loss

    #discriminator loss is the binary cross entropy loss of classification
    disc_T1_loss = discriminator_loss(disc_real_T1, disc_synthetic_T1)
    disc_T2_loss = discriminator_loss(disc_real_T2, disc_synthetic_T2)
    
  generator_T1T2_gradients = tape.gradient(total_generator_T1T2_loss, 
                                        generator_T1T2.trainable_variables)
  generator_T2T1_gradients = tape.gradient(total_generator_T2T1_loss, 
                                        generator_T2T1.trainable_variables)
  
  discriminator_T1T2_gradients = tape.gradient(disc_T1_loss, 
                                            discriminator_T1T2.trainable_variables)
  discriminator_T2T1_gradients = tape.gradient(disc_T2_loss, 
                                            discriminator_T2T1.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_T1T2_optimizer.apply_gradients(zip(generator_T1T2_gradients, 
                                            generator_T1T2.trainable_variables))

  generator_T2T1_optimizer.apply_gradients(zip(generator_T2T1_gradients, 
                                            generator_T2T1.trainable_variables))
  
  discriminator_T1T2_optimizer.apply_gradients(zip(discriminator_T1T2_gradients,
                                                discriminator_T1T2.trainable_variables))
  
  discriminator_T2T1_optimizer.apply_gradients(zip(discriminator_T2T1_gradients,
                                                discriminator_T2T1.trainable_variables))
