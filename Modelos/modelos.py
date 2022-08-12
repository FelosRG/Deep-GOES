import tensorflow as tf
from  tensorflow.keras import layers


normalizacion = {
    "4":(0,20),
    "6":(0,9),
    "8":(0.5,3),
    "9":(1.8,8.2),
    "10":(2.2,16),
    "14":(20,130),
    "16":(30,115),
    "CM":(0,1),
    "CTH":(0,14_500),
    "COD":(0,30),
    "CAPE":(0,250),
    "CTP":(0,4),
    "Altura":(0,3500),
}

class Modelo_CBM(tf.keras.Model):
    """
    -------------------
    -Cloud Binary Mask-
    -------------------

    Modelo para la clasificación de  nubes y no  nubes.
    Dado que es un tipo  de "segmentación de imágen" la
    salida son dos canales, uno para la probabilidad de
    nube y otro para probabilidad de no nube.

    * Entrenamiento:
        X: (batch_size,37,37,5)
        y: (batch_size,37,37,2)

        Con los canales de X:
        [4,6,14,16,Altura]
        Con los canales de y:
        [prob_nube,prob_clearsky]
    """

    def __init__(self,**kwards):
        super().__init__(**kwards)

        self.train = True
        self.class_weight = {0:1,1:0.383}

        # [Primera parte densa]
        self.denso1 = layers.Dense(10,activation="relu",input_shape=(37,37,5))
        self.denso2 = layers.Dense(10,activation="relu")
        self.denso3 = layers.Dense(10,activation="relu")

        # [Segunda parte convolucional]
        self.batch1 = layers.BatchNormalization()
        self.conv1  = layers.Conv2D(15,3 ,padding="same" ,activation="relu"   )
        self.batch2 = layers.BatchNormalization()
        self.conv2  = layers.Conv2D(15,3 ,padding="same" ,activation="relu"   )
        self.batch3 = layers.BatchNormalization()
        self.conv3  = layers.Conv2D(2 ,3 ,padding="same" ,activation="softmax")
    
    def call(self,inputs):
        # [Primera parte densa]
        output = self.denso1(inputs)
        output = self.denso2(output)
        output = self.denso3(output)

        # [Segunda parte convolucional]
        output = self.batch1(output)
        output = self.conv1(output)
        output = self.batch2(output)
        output = self.conv2(output)
        output = self.batch3(output)
        output = self.conv3(output)
        
        # Cuando no se esté en entrenamiento solo se devuelve
        # la probabilidad de nube.
        if self.train == False:
            output = output[:,:,:,0]

        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(37, 37, 5))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Modelo_CTP(tf.keras.Model):
    """
    -------------------
    -Cloud Top Phase-
    -------------------

    Modelo para la clasificación del tipo
    de la nube.


    """
    def __init__(self,**kwards):
        super().__init__(**kwards)
        self.class_weigts = {0:0.159,1:0.122,2:0.716,3:1,4:0.0578}



class Modelo_CTH(tf.keras.Model):
    """
    -------------------
    -Cloud Top Height-
    -------------------

    Modelo para la estimación de la altura del tope
    de las nubes.

    * Entrenamiento:
        X: (batch_size,37,37,6)
        y: (batch_size,37,37,2)

        Con los canales de X:
        [4,6,14,16,Altura,CM]

        Con los canales de y:
        [CTH]
    """

    def __init__(self):
        super().__init__(name="CloudBinaryMask")

        # [Primera parte densa]
        self.denso1 = layers.Dense(15,activation="tanh",input_shape=(37,37,5))
        self.denso2 = layers.Dense(10 ,activation="relu")
        self.denso3 = layers.Dense(10 ,activation="relu")
        self.denso4 = layers.Dense(1,activation="relu")

    def call(self,inputs):
        # Extraemos el cloud mask del resto de los datos.
        cm = inputs[:,:,:,-1]
        cm = tf.expand_dims(cm,axis=3)

        inputs = inputs[:,:,:,:-1]

        output = self.denso1(inputs)
        output = self.denso2(output)
        output = self.denso3(output)
        output = self.denso4(output)

        # Aplicamos el cloud mask.
        output = cm*output
        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(37, 37, 6))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Modelo_CAPE(tf.keras.Model):
    """
    -------------------
    -CAPE-
    -------------------

    Modelo para la estimación del CAPE de
    la atmósfera.

    * Entrenamiento:
        X: (batch_size,37,37,6)
        y: (batch_size,37,37,2)

        Con los canales de X:
        [8,9,10,14,16,Altura,CM]

        Con los canales de y:
        [CAPE]
    """

    def __init__(self):
        super().__init__(name="CAPE")

        # [Primera parte densa]
        self.denso1 = layers.Dense(15,activation="relu",input_shape=(37,37,6))
        self.denso2 = layers.Dense(15,activation="relu")
        self.denso3 = layers.Dense(15,activation="relu")
        self.denso4 = layers.Dense(15,activation="relu")
        self.conv1  = layers.Conv2D(15,3 ,padding="same" ,activation="relu"   )
        self.conv2  = layers.Conv2D(15,3 ,padding="same" ,activation="relu"   )
        self.conv3  = layers.Conv2D(15,3 ,padding="same" ,activation="relu"   )
        self.denso5 = layers.Dense(1,activation="relu")
    def call(self,inputs):
        # Extraemos el cloud mask del resto de los datos.

        cm = inputs[:,:,:,-1]
        cm = tf.expand_dims(cm,axis=3)
        cm = tf.math.abs(cm-1)

        output = self.denso1(inputs)
        output = self.denso2(output)
        output = self.denso3(output)
        output = self.denso4(output)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.denso5(output)
        return output*cm
    
    def summary(self):
        x = tf.keras.Input(shape=(37, 37, 7))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

class Modelo_COD(tf.keras.Model):
    """
    -------------------
    -Cloud Optica Depth-
    -------------------

    Modelo para la estimación de la opacidad
    de las nubes.

    * Entrenamiento:
        X: (batch_size,37,37,6)
        y: (batch_size,37,37,2)

        Con los canales de X:
        [4,6,14,16,Altura,CM]

        Con los canales de y:
        [CTH]
    """

    def __init__(self):
        super().__init__(name="CloudOpticalDepth")

        # [Primera parte densa]
        self.denso1 = layers.Dense(15 ,activation="tanh",input_shape=(37,37,5),name="Denso1")
        self.denso2 = layers.Dense(15 ,activation="relu",name="Denso2")
        self.denso3 = layers.Dense(10 ,activation="relu",name="Denso3")
        self.conv1  = layers.Conv2D(20,3,activation="relu",padding="same")
        self.conv2  = layers.Conv2D(20,3,activation="relu",padding="same")
        self.denso = layers.Dense(1 ,activation="relu",name="Denso4")

    def call(self,inputs):
        # Extraemos el cloud mask del resto de los datos.
        cm = inputs[:,:,:,-1]
        cm = tf.expand_dims(cm,axis=3)
        inputs = inputs[:,:,:,:-1]
        
        # Operaciones de la red
        output = self.denso1(inputs)
        output = self.denso2(output)
        output = self.denso3(output)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.denso(output)

        # Aplicamos el cloud mask.
        output = cm*output

        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(37, 37, 6))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class GenBlock(layers.Layer):
    def __init__(self,canales_output,kernel_size=4,strides=2,padding="same",**kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = layers.Conv2DTranspose(canales_output,kernel_size,strides,use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.ReLU()
    def call(self,X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))

class DisBlock(layers.Layer):
    def __init__(self,canales_output,kernel_size=4,padding="same",alpha=0.2,**kwargs):
        super().__init__(**kwargs)
        self.conv2d  = layers.Conv2D(canales_output,kernel_size,padding=padding,use_bias=False)
        self.maxpool = layers.MaxPooling2D()
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha)
    def call(self,X):
        return self.batch_norm(self.maxpool(self.activation(self.conv2d(X))))

class Modelo_GAN_generador(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        
        self.num_inputs = 50
        
        # Padding
        self.padding = layers.ZeroPadding2D(padding=((0,1),(0,1)),name="padding")
        
        # Capas
        self.denso_in = layers.Dense(9*9*self.num_inputs,activation="relu")
        self.deconv1  = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation="relu",use_bias=False)
        self.batch1   = layers.BatchNormalization()
        self.deconv2  = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation="relu",use_bias=False)
        self.batch2   = layers.BatchNormalization()
        self.conv_out1 = layers.Conv2D(16,(9, 9), padding="same", activation = "relu")
        self.conv_out2 = layers.Conv2D(3, (3, 3), padding="same", activation = "tanh")
        
    def call(self,inputs):
        output = self.denso_in(inputs)
        output = tf.reshape(output,(-1,9,9,self.num_inputs))
        output = self.deconv1(output)
        output = self.batch1(output)
        output = self.deconv2(output)
        output = self.batch2(output)
        output = self.padding(output)
        output = self.conv_out1(output)
        output = self.conv_out2(output)
        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(self.num_inputs))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
class Modelo_GAN_discriminador(tf.keras.Model):
    def __init__(self,):
        super().__init__()

        self.num_inputs = 50
        
        # Función de activción
        self.lrelu1 = layers.LeakyReLU()
        
        self.conv1    = layers.Conv2D(32 , (5, 5), strides=(2, 2), padding="same",activation="linear")
        self.lrelu1   = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)
        self.conv2    = layers.Conv2D(64 , (5, 5), strides=(2, 2), padding="same",activation="linear")
        self.lrelu2   = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)
        self.flatten  = layers.Flatten()
        self.denso    = layers.Dense(1)
        
    def call(self,inputs):
        output = self.conv1(inputs)
        output = self.lrelu1(output)
        output = self.dropout1(output)
        output = self.conv2(output)
        output = self.lrelu2(output)
        output = self.dropout2(output)
        output = self.flatten(output)
        output = self.denso(output)
        return output

    def summary(self):
        x = tf.keras.Input(shape=(37,37,3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
class Modelo_Encoder(tf.keras.Model):
    def __init__(self,num_inputs):
        super().__init__()
        self.block1 = DisBlock(64)
        self.block2 = DisBlock(32)
        self.block3 = DisBlock(16)
        self.flatten = layers.Flatten()
        self.denso  = layers.Dense(num_inputs,activation="linear")
    def call(self,X):
        # Input (None,37,37,3)
        output = self.block1(X) # Output (None,18,18,64)
        output = self.block2(output)  # Output (None,9,9,32)
        output = self.block3(output)  # Output (None,4,4,16)
        output = self.flatten(output) # Output (None,256)
        output = self.denso(output)   # Output (None,50)
        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(37,37,3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

class Modelo_Autoencoder(tf.keras.Model):
    def __init__(self,num_inputs,path_pesos_decoder):
        super().__init__()
        self.encoder = Modelo_Encoder(num_inputs)
        self.decoder = Modelo_GAN_generador()
        # Cargamos y congelamos los pesos del decoder.
        self.decoder.load_weights(path_pesos_decoder)
        self.decoder.trainable = False
    def call(self,X):
        lat_space = self.encoder(X)
        output = self.decoder(lat_space)
        return output
    
    def summary(self):
        x = tf.keras.Input(shape=(37,37,3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

if __name__ == "__main__":
    encoder = Modelo_Encoder(num_inputs=50,path_pesos_decoder="")
    encoder.summary()
