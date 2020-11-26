#contains the GAN discriminator


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

from IPython.display import clear_output

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])

  downsampler_1 = tf.keras.Sequential([
   tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.LeakyRELU()
  ])

  downsampler_2 = tf.keras.Sequential([
   tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.LeakyRELU()
  ])

    downsampler_3 = tf.keras.Sequential([
   tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.LeakyRELU()
  ])

  down1 = down_sampler1(x) 
  down2 = down_sampler2(down1)
  down3 = down_sampler3(down2)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



