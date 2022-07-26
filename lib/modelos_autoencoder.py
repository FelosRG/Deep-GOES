import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras.layers import Conv2D ,MaxPooling2D ,UpSampling2D, ZeroPadding2D

import os
_root_path = os.getcwd()
_root_path = "/".join(_root_path.split("/")[:-1])

import numpy as np

class Acentuador(keras.Model):
    """
    Modelo que realiza la tarea de ajustar la salida
    del autoencoder.

    Input : Array con shape (-1,31,31,1)
    Output: Array con shape (-1,31,31,1)
    """

    def __init__(self):
        super().__init__(name="Acentuador")
        self.path_pesos = _root_path + "/Pesos/Acentuador.hdf5"
        self.conv1   = Conv2D(6,kernel_size=5,activation="relu"   ,strides= 1,padding="same",name="conv1")
        self.conv2   = Conv2D(12,kernel_size=3,activation="relu"   ,strides= 1,padding="same",name="conv2")
        self.conv3   = Conv2D(20,kernel_size=3,activation="relu"   ,strides= 1,padding="same",name="conv3")
        self.conv4   = Conv2D(1,kernel_size=3,activation="sigmoid",strides= 1,padding="same",name="conv4")
        
    def call(self,X):
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output

    def cargar_pesos(self):
        try:
            X = np.random.rand(10,31,31,1)
            _ = self.call(X)
            self.load_weights(self.path_pesos)
        except FileNotFoundError:
            print("Archivo de cargado no encontrado.")


class Encoder(keras.Model):
    def __init__(self,filtros1,filtros2,filtros_encoder):
        super().__init__(name="Encoder")
        self.f1 = filtros1
        self.f2 = filtros2
        self.fe = filtros_encoder
        self.encoder_11   = Conv2D(self.f1,kernel_size=5,activation= "relu",strides= 1,padding="same" ,name="encoder_11")
        self.encoder_12   = Conv2D(self.f1,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="encoder_12")
        self.encoder_13   = Conv2D(self.f1,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="encoder_13")
        self.maxPooling_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid",name="maxPooling_1")
        self.encoder_21   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="encoder_21")
        self.encoder_22   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="encoder_22")
        self.encoder_23   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="encoder_23")
        self.maxPooling_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid",name="maxPooling_2")
        self.combinador   = Conv2D(self.fe,kernel_size=1,activation="tanh",padding="same",name="combinador")
        
    def call(self,X):
        output = self.encoder_11(X)
        output = self.encoder_12(output)
        output = self.encoder_13(output)
        skip = self.maxPooling_1(output) # Tras el primer pooling pasa a 15,15
        output = self.encoder_21(skip)
        output = self.encoder_22(output)
        output = self.encoder_23(output)
        output = self.maxPooling_2(output)
        output = self.combinador(output)
        return output #, skip


class Decoder(keras.Model):
    def __init__(self,filtros1,filtros2):
        super().__init__(name="Decoder")
        self.f1 = filtros1
        self.f2 = filtros2
        #self.ajustador_1 = Conv2D(self.f2,kernel_size=3,activation="relu",padding="valid",name="ajustador_skip_1")
        #self.ajustador_2 = Conv2D(self.f2,kernel_size=1,activation="relu",padding="same" ,name="ajustador_skip_2")
        #self.padding_1   = ZeroPadding2D(padding=((0,1),(0,1)),name="padding_1")
        self.upSampling_1 = UpSampling2D(size=(2,2))
        self.decoder_11   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same",name="decoder_11")
        self.decoder_12   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same",name="decoder_12")
        self.decoder_13   = Conv2D(self.f2,kernel_size=3,activation= "relu",strides= 1,padding="same",name="decoder_13")
        self.upSampling_2 = UpSampling2D(size=(2,2))
        self.padding_2    = ZeroPadding2D(padding=((1,2),(1,2)),name="padding_2")
        self.decoder_21   = Conv2D(self.f1,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="decoder_21")
        self.decoder_22   = Conv2D(self.f1,kernel_size=3,activation= "relu",strides= 1,padding="same" ,name="decoder_22")
        self.decoder_23   = Conv2D(self.f1,kernel_size=5,activation= "relu",strides= 1,padding="same" ,name="decoder_23")
        self.combinador = Conv2D(1,kernel_size=1,activation="tanh",padding="same",name="combinador")
        
    def call(self,Input):
        #X ,skip = Input
        X = Input
        #skip = self.ajustador_1(skip)
        #skip = self.ajustador_2(skip)
        #skip = self.padding_1(skip)
        output = self.upSampling_1(X)
        output = self.decoder_11(output)
        output = self.decoder_12(output) #+ skip # -> Conectamos skip connection
        output = self.decoder_13(output) 
        output = self.upSampling_2(output)
        output = self.padding_2(output)
        output = self.decoder_21(output)
        output = self.decoder_22(output)
        output = self.combinador(output)
        return output

class Autoencoder(keras.Model):
    def __init__(self):
        super().__init__(name="Autoencoder")
        self.filtros1 = 6
        self.filtros2 = 12
        self.filtros_encoder = 3
        self.encoder = Encoder(self.filtros1,self.filtros2,self.filtros_encoder)
        self.decoder = Decoder(self.filtros1,self.filtros2)
    def call(self,X):
        #X,skip = self.encoder(X)
        #output = self.decoder((X,skip))
        X = self.encoder(X)
        output = self.decoder(X)
        return output

    def cargar_pesos(self):
        X = np.random.rand(10,31,31,1)
        _ = self.call(X)
        self.load_weights(_root_path + "/Pesos/Autoencoder.hdf5")

    