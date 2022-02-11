'''
Filipe Chagas
11 - Feb - 2022
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm
import os
from typing import *

def contrastive_loss(y: tf.Tensor, preds: tf.Tensor, margin=1) -> tf.Tensor:
    '''
    Loss function of the SNN.
    Ref: https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
    '''
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
	
    # calculate the contrastive loss between the true labels and
	# the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	
    # return the computed contrastive loss to the calling function
    return loss

def euclidean_distance(vectors: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    '''
    Euclidean distance between two vectors.
    Ref: https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
    
    Args:
        vectors (Tuple[tf.Tensor,tf.Tensor]): Tuple with two tensorflow vectors.
    Returns (tf.Tensor): Enclidean distance. 
    '''
	# unpack the vectors into separate lists
    (featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
	
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

class SNNGenerator():  
    """
    Abstract data generator class for SNN
    """ 
    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        raise Exception('The SNNGerator class is abstract. To be usable, it must be inherited and the __len__ and __getitem__ methods must be overridden.')
        #return 0

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        raise Exception('The SNNGerator class is abstract. To be usable, it must be inherited and the __len__ and __getitem__ methods must be overridden.')
        #return ([None, None], None)

class SNN():
    def __init__(self, input_shape, encoder: keras.Model, optimizer: optimizers.Optimizer = 'adamax', distance_function: Callable = euclidean_distance):
        """Generate a SNN arch for the encoder model.
        Args:
            encoder_model (keras.Model): Uncompiled model that will be used as encoder.
            optimizer (optimizers.Optimizer, optional): Keras optimizer that will be used. Defaults to 'adam'.
            distance_function (Callable): A distance function in the format *d(vectors: Tuple[tf.Tensor, tf.Tensor])->tf.Tensor*
        """
        self.__encoder__ = encoder
        self.__optimizer__ = optimizer
        self.__distance_function__ = distance_function
        self.training_loss_history = []
        self.validation_loss_history = []

        #SNN has two inputs.
        #input_a = layers.Input(shape=encoder.layers[0].input_shape[1:])
        #input_b = layers.Input(shape=encoder.layers[0].input_shape[1:])
        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)

        #One encoder for each input. Both encoders have the same weights.
        encoder_a = encoder(input_a)
        encoder_b = encoder(input_b)

        #Join encoders to a distance layer
        snn_output = layers.Lambda(distance_function, name='Distance')([encoder_a, encoder_b])

        #Turn tensors to a keras compiled model.
        self.keras_model = keras.Model(inputs=[input_a, input_b], outputs=snn_output)
        self.keras_model.compile(loss=contrastive_loss, optimizer=optimizer)
    
    def get_encoder(self) -> keras.Model:
        return self.__encoder__

    def fit(self, training_generator: SNNGenerator, validation_generator: SNNGenerator, epochs: int, start_epoch: int = 1, epoch_end_callback = None) -> Tuple[List[float], List[float]]:
        """This function is a substitute for the KERAS MODEL.FIT method. 
        Justification - I tried to use the Model.fit method with SNNGenerator (inheriting the Sequence class), but it does not work and I do not know what causes the problem.
        
        Args:
            training_generator (SNNGenerator): Training data generator object.
            validation_generator (SNNGenerator): Validation data generator object.
            epochs (int): Number of training epochs.
            start_epoch (int, optional): Epoch in which training should begin. Defaults to 1.
            epoch_end_callback (function, optional): Function that must be called at each epoch end receiving as arguments: snn, epoch, training_loss and validation_loss. Defaults to None.
        
        Return (Tuple[List[float], List[float]]): training_loss_history and validation_loss_history.
        """
        assert epochs >= 1
        assert start_epoch >= 1
        assert epochs >= start_epoch
        assert len(training_generator) > len(validation_generator)

        self.training_loss_history = []
        self.validation_loss_history = []
        
        for epoch in range(start_epoch-1, epochs): #For each epoch
            print(f'EPOCH {epoch+1} OF {epochs}')
            
            training_loss_sum = 0 #This variable will accumulate sums of the loss of each step of train_on_batch
            validation_loss_sum = 0 #This variable will accumulate sums of the loss of each step of eval-on-batch
            training_loss = 0 #This variable will be updated to each iteration with the mean of the loss of each step of train_on_batch
            validation_loss = 0 #This variable will be updated to each iteration with the mean of the loss of each step of eval-on-batch

            # --- Epoch training loop ---
            for i in tqdm(range(len(training_generator))): #For each batch index
                x, y = training_generator[i] #Get the current training batch
                training_loss_sum += self.keras_model.train_on_batch(x, y, return_dict=True)['loss'] #Update models's weights for the current batch
                training_loss = training_loss_sum / (i+1) #Update loss mean

            print(f'training_loss = {training_loss:.4f}')
            self.training_loss_history.append(training_loss)

            # --- Epoch validation loop ---
            for i in tqdm(range(len(validation_generator))): #For each batch index
                vx, vy = validation_generator[i] #Get the current validation batch
                validation_loss_sum += self.keras_model.evaluate(vx, vy, batch_size=validation_generator.__batch_size__, verbose=0, return_dict=True)['loss'] #evaluate model with the current batch
                validation_loss = validation_loss_sum / (i+1) #Update loss mean

            print(f'validation_loss={validation_loss:.4f}')
            self.validation_loss_history.append(validation_loss)

            # ---  End of the epoch ---
            training_generator.on_epoch_end()

            if epoch_end_callback != None:
                epoch_end_callback(snn=self, epoch=epoch, training_loss=training_loss, validation_loss=validation_loss)
    
    def save_encoder(self, path: str):
        self.__encoder__.save_weights(path)
    
    def load_encoder(self, path: str):
        self.__encoder__.load_weights(path)