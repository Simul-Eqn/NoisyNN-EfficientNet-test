import sys
if 'idlelib.run' in sys.modules: sys.path.pop(1) # fixes an issue with import 
sys.path.insert(0, '../')

import os
os.environ['KERAS_BACKEND'] = "tensorflow" 


import tensorflow as tf 
import keras 

import numpy as np 

from data_generation import StanfordCarsDataloader as SCDL
from tf_noisynn import LinearTransformNoiseLayer as LTNL 
from tests.efficientnet import * 

target_img_shape = (224, 224, 3) 
dataloader = SCDL('train', data_shape=target_img_shape) 





dnn_params = {
    'depth_coefficient': 1.0, 
    'width_coefficient': 1.0, 
    'drop_connect_rate': 0.2, 
}
dropout_rate = 0.2 



noiseless_model = keras.Sequential([
    keras.layers.Input(dataloader.data_shape),

    EfficientNetItems.get_conv1(**dnn_params), 
    EfficientNetItems.get_bn1(), 
    keras.activations.swish, 

    *EfficientNetItems.get_block1(**dnn_params), 
    *EfficientNetItems.get_block2(**dnn_params), 
    *EfficientNetItems.get_block3(**dnn_params), 
    *EfficientNetItems.get_block4(**dnn_params), 
    *EfficientNetItems.get_block5(**dnn_params), 
    *EfficientNetItems.get_block6(**dnn_params), 
    *EfficientNetItems.get_block7(**dnn_params), 

    EfficientNetItems.get_conv2(**dnn_params), 
    EfficientNetItems.get_bn2(), 
    keras.activations.swish, 
    EfficientNetItems.get_pool(), 
    EfficientNetItems.get_dropout(dropout_rate), 
    EfficientNetItems.get_fc(196), 
 
    ])

ltnl_model = keras.Sequential([
    keras.layers.Input(dataloader.data_shape),

    keras.layers.Conv2D(32, 3, padding='same', activation=None), 
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Activation('relu'),
    
    keras.layers.Conv2D(20, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(pool_size=3),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(10, 5, padding='same', activation=None),
    keras.layers.MaxPooling2D(pool_size=5),
    keras.layers.Activation('relu'),
    
    keras.layers.Flatten(),
    LTNL(), 

    keras.layers.Dense(640, activation='relu'), 
    LTNL(), 

    keras.layers.Dense(320,  activation='relu'),
    LTNL(), 
    
    keras.layers.Dense(196, activation='relu'), 
    ])


def test_noiseless_model(): 
    print("SUMMARY OF EfficientNetB0 (no noise injected) MODEL: ")
    noiseless_model.summary() 

    noiseless_model.compile(optimizer='adam',
                loss = keras.losses.BinaryCrossentropy(from_logits=True),
                metrics = ['accuracy', 'precision', 'recall']) 
    
    # TODO: TEST IT 



def test_ltnl_model(): 
    print("SUMMARY OF EfficientNetB0 + LTNL MODEL: ")
    ltnl_model.summary() 

    ltnl_model.compile(optimizer='adam',
                loss = keras.losses.BinaryCrossentropy(from_logits=True),
                metrics = ['accuracy', 'precision', 'recall']) 
    
    # TODO: TEST IT 

