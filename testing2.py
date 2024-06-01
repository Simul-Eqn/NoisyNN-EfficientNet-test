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

'''
print() 
print(*[
    keras.layers.Input(dataloader.data_shape),

    EfficientNetItems.get_conv1(**dnn_params), 
    EfficientNetItems.get_bn1(), 
    keras.layers.Activation(keras.activations.swish), 

    *EfficientNetItems.get_block1(**dnn_params), 
    *EfficientNetItems.get_block2(**dnn_params), 
    *EfficientNetItems.get_block3(**dnn_params), 
    *EfficientNetItems.get_block4(**dnn_params), 
    *EfficientNetItems.get_block5(**dnn_params), 
    *EfficientNetItems.get_block6(**dnn_params), 
    *EfficientNetItems.get_block7(**dnn_params), 

    EfficientNetItems.get_conv2(**dnn_params), 
    EfficientNetItems.get_bn2(), 
    keras.layers.Activation(keras.activations.swish), 
    EfficientNetItems.get_pool(), 
    EfficientNetItems.get_dropout(dropout_rate), 
    EfficientNetItems.get_fc(196), 
 
    ], sep='\n')
print()
'''

noiseless_model = keras.Sequential([
    keras.layers.Input(dataloader.data_shape),

    EfficientNetItems.get_conv1(**dnn_params), 
    EfficientNetItems.get_bn1(), 
    keras.layers.Activation(keras.activations.swish), 

    *EfficientNetItems.get_block1(**dnn_params), 
    *EfficientNetItems.get_block2(**dnn_params), 
    *EfficientNetItems.get_block3(**dnn_params), 
    *EfficientNetItems.get_block4(**dnn_params), 
    *EfficientNetItems.get_block5(**dnn_params), 
    *EfficientNetItems.get_block6(**dnn_params), 
    *EfficientNetItems.get_block7(**dnn_params), 

    EfficientNetItems.get_conv2(**dnn_params), 
    EfficientNetItems.get_bn2(), 
    keras.layers.Activation(keras.activations.swish), 
    EfficientNetItems.get_pool(), 
    EfficientNetItems.get_dropout(dropout_rate), 
    EfficientNetItems.get_fc(196), 
 
    ])

ltnl_model = keras.Sequential([
    keras.layers.Input(dataloader.data_shape),

    EfficientNetItems.get_conv1(**dnn_params), 
    EfficientNetItems.get_bn1(), 
    keras.layers.Activation(keras.activations.swish), 

    LTNL(), 

    *EfficientNetItems.get_block1(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block2(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block3(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block4(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block5(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block6(**dnn_params), 
    LTNL(), 
    *EfficientNetItems.get_block7(**dnn_params), 
    LTNL(), 

    EfficientNetItems.get_conv2(**dnn_params), 
    EfficientNetItems.get_bn2(), 
    keras.layers.Activation(keras.activations.swish), 
    LTNL(), 
    EfficientNetItems.get_pool(), 
    EfficientNetItems.get_dropout(dropout_rate), 
    EfficientNetItems.get_fc(196), 
    ])


def test_noiseless_model(): 
    print("SUMMARY OF EfficientNetB0 (no noise injected) MODEL: ")
    noiseless_model.summary() 

    noiseless_model.compile(optimizer='adam',
                loss = keras.losses.BinaryCrossentropy(from_logits=True),
                metrics = ['accuracy', 'precision', 'recall']) 

    import time
    start_time = time.time() 
    res = (noiseless_model(dataloader[0][0]))
    end_time = time.time()
    print("GOTTEN RES:", res)
    print("TIME TAKEN:", end_time-start_time) 
    # TODO: TEST IT 



def test_ltnl_model(): 
    print("SUMMARY OF EfficientNetB0 + LTNL MODEL: ")
    ltnl_model.summary() 

    ltnl_model.compile(optimizer='adam',
                loss = keras.losses.BinaryCrossentropy(from_logits=True),
                metrics = ['accuracy', 'precision', 'recall']) 

    import time
    start_time = time.time() 
    res = (ltnl_model(dataloader[0][0]))
    end_time = time.time()
    print("GOTTEN RES:", res)
    print("TIME TAKEN:", end_time-start_time) 
    # TODO: TEST IT

#test_noiseless_model() 

#test_ltnl_model()


def test_time_of(model, n:int=20):
    import time

    model(dataloader[0][0]) 
    
    start_time = time.time() 
    for i in range(n):
        res = model(dataloader[i][0])
    end_time = time.time()
    print("TIME TAKEN FOR "+str(n)+" TESTS:", end_time-start_time) 


