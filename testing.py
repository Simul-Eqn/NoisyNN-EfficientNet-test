import sys
if 'idlelib.run' in sys.modules: sys.path.pop(1) # fixes an issue with import 
sys.path.append('./data_generation') 

import os
os.environ['KERAS_BACKEND'] = "tensorflow" 


import tensorflow as tf 
import keras 

import numpy as np 

from stanford_cars_dataloader import StanfordCarsDataloader as SCDL


model = keras.Sequential([
    keras.layers.Conv2D(10, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(),
    keras.layers.Activation('relu'), 
    keras.layers.Conv2D(20, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(),
    keras.layers.Activation('relu'), 
    keras.layers.Flatten(),
    keras.layers.Dense(200, input_shape=100, activation='swish'),
    keras.layers.Dense(196, input_shape=100, activation='swish'), 
    ])

model.compile(optimizer='adam',
              loss = keras.losses.BinaryCrossentropy(from_logits=True),
              metrics = ['accuracy', 'precision', 'recall']) 

