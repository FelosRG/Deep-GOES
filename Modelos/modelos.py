import tensorflow as tf
from  tensorflow.keras import layers


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
        X: (batch_size,29,29,5)
        y: (batch_size,29,29,2)

        Con los canales de X:
        [4,6,14,16,Altura]
        Con los canales de y:
        [prob_nube,prob_clearsky]
    """

    def __init__(self):
        super().__init__(name="CloudBinaryMask")

        self.train = True

        # [Primera parte densa]
        self.denso1 = layers.Dense(5,activation="relu",input_shape=(29,29,5))
        self.denso2 = layers.Dense(5,activation="relu")
        self.denso3 = layers.Dense(5,activation="relu")

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
        x = tf.keras.Input(shape=(29, 29, 5))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

if __name__ == "__main__":
    modelo_CBM = Modelo_CBM()
    modelo_CBM.train = True
    modelo_CBM.summary()
