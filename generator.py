
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

from IPython.display import clear_output

OUTPUT_CHANNELS = 3

initializer = tf.random_normal_initializer(0,0.02)

def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.LeakyRELU()
  ]), 
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]),
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
        tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
        tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
        tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
        tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]), 
  ]

  up_stack = [
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]),
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]),
   tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]),
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   result.add(tf.keras.layers.Dropout(0.5)),
   tf.keras.layers.LeakyRELU()
  ]),
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.LeakyRELU()
  ]),
    tf.keras.Sequential([
   tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.LeakyRELU()
  ]),
     tf.keras.Sequential([
   tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False),
   tf.keras.layers.LeakyRELU()
  ])
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2, padding='same', kernel_initializer=initializer,
                                         activation='tanh')

  #residual connections for the neural network
  x = inputs
  skips = []
  for down in down_stack:
    x = down(x); skips.append(x)
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = tf.keras.layers.Concatenate()([up(x), skip])
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)
