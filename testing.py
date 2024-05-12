import sys
if 'idlelib.run' in sys.modules: sys.path.pop(1) # fixes an issue with import 
sys.path.append('./data_generation') 

import os
os.environ['KERAS_BACKEND'] = "tensorflow" 


import tensorflow as tf 
import keras 

import numpy as np 

from stanford_cars_dataloader import StanfordCarsDataloader as SCDL

target_img_shape = (240, 360, 3) 


dataloader = SCDL('train', data_shape=target_img_shape) 


model = keras.Sequential([
    keras.layers.Input(dataloader.data_shape),
    
    keras.layers.Conv2D(10, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Activation('relu'),
    
    keras.layers.Conv2D(20, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(pool_size=3),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(10, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(pool_size=5),
    keras.layers.Activation('relu'),
    
    keras.layers.Flatten(),

    keras.layers.Dense(640, activation='relu'), 

    keras.layers.Dense(320,  activation='relu'),
    
    keras.layers.Dense(196, activation='relu'), 
    ])

model.summary() 

model.compile(optimizer='adam',
              loss = keras.losses.BinaryCrossentropy(from_logits=True),
              metrics = ['accuracy', 'precision', 'recall']) 

