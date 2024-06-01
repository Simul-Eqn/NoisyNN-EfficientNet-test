import keras 

class SaveEveryEpochCallback(keras.callbacks.Callback): 

    def __init__(self, path_prefix): 
        super().__init__() 
        self.path_prefix = path_prefix # PATH_PREFIX ALSO CONTAINS THE SLASH, IS A STRING 

    def on_epoch_end(self, epoch, logs): 
        self.model.save(self.path_prefix+"epoch_"+str(epoch)+".keras")


def get_valid_samplers(k=5): # TODO: MAKE THE SAMPLERS 
    for i in range(k): 
        def sampler(indices): 
            pass 
    return [] 

