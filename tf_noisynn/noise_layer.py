import tensorflow as tf 
import keras 
import math 

#tf.debugging.set_log_device_placement(True)

warn = True 

def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input. 
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = tf.linalg.diag(tf.ones(k))
    shift_identity = tf.zeros(k, k) 
    for i in range(k):
        shift_identity[(i+1)%k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt

def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input. 
    Suppose 1_(kxk) is tf.ones
    """
    return tf.linalg.diag(tf.ones(k)) * -k/(k+1) + tf.ones((k, k)) / (k+1)




def get_linear_transform_noise_of(x:tf.Tensor, noise_mat=None, ): 
    assert len(x.shape)==2, "Output must be batched one-dimensional data. "
    if (x.dtype != tf.float32): 
        if warn: 
            print("\nWARNING: DType of linear transform noise input is not float32. Attempting typecast (no error means success)\n")
        x = tf.cast(x, tf.float32) 

    if (noise_mat != None): assert len(noise_mat.shape)==2, "Noise matrix must be a 2-dimensional matrix" 

    if (noise_mat==None): noise_mat = optimal_quality_matrix(x.shape[-1]) # choose optimal quality matrix by default 
    
    return x@noise_mat # matrix multiply them 

def add_linear_transform_noise_to(x:tf.Tensor, noise_mat=None, ): 
    return get_linear_transform_noise_of(x, noise_mat) + x 


class LinearTransformNoiseLayer(keras.layers.Layer): 
    def __init__(self, noise_mat=None, ): 
        super(LinearTransformNoiseLayer, self).__init__() 
        self.noise_mat = noise_mat 

    def build(self, input_shape): 
        if (self.noise_mat == None): 
            self.noise_mat = optimal_quality_matrix(input_shape[-1]) 
    
    def call(self, x): 
        return add_linear_transform_noise_to(x, self.noise_mat) 
    



