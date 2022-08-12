import os
import sys
import h5py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils

DIR_SCRIPT  = "/".join(os.path.realpath(__file__).split("/")[:-1])
DIR_REPO    = "/".join(DIR_SCRIPT.split("/")[:-2])
DIR_LIB     = f"{DIR_REPO}/lib"
DIR_MODELOS = f"{DIR_REPO}/Modelos" 
PATH_DATASET  = f"{DIR_REPO}/gendata/Datasets/dataset.h5"
PATH_DATASET_TRAIN = f"{DIR_REPO}/gendata/Datasets/dataset_train.h5"
PATH_DATASET_TEST  = f"{DIR_REPO}/gendata/Datasets/dataset_test.h5"
PATH_DATASET_GAN   = f"{DIR_REPO}/gendata/Datasets/dataset_GAN.h5"
DIR_PESOS  = f"{DIR_REPO}/Modelos/Pesos/"
PATH_DIC_NORM = f"{DIR_PESOS}/norm.dic"

sys.path.append(DIR_LIB)
sys.path.append(DIR_MODELOS)
import datasets
import modelos

BATCH_SIZE = 256
TEST_FRAC  = 0.1

# Extraemos el diccionario de normalización
DIC_NORM = modelos.normalizacion

# Dimención del ruido de la red GAN
GAN_NOISE_DIM = 50

def ver_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print("Name:", gpu.name, "  Type:", gpu.device_type)

def preparar_dataset():
    # Revisamos que ya exitan los archivos.
    if os.path.isfile(PATH_DATASET_TRAIN) or os.path.isfile(PATH_DATASET_TEST):
        print("Ya existen los datasets preparados,si desea volver a generarlos eliminelos y vulva a ejecutar la función.") 
    else:
        # Caracterizando el dataset.
        with h5py.File(PATH_DATASET,"r") as file:
            lista_keys = list(file.keys())
            num_archivos = file[lista_keys[0]].shape[0]
            num_ubicaciones = file[lista_keys[0]].shape[1]
            resolucion = file[lista_keys[0]].shape[2]

        # Generando indice de separación de train y test.
        num_datos = num_archivos*num_ubicaciones
        idx_sep   = np.floor(num_datos*TEST_FRAC).astype(int)

        # Randomizando orden de los índices
        idx_rand = np.arange(num_datos)
        np.random.shuffle(idx_rand)

        # Proceso principal
        dic_norm = {}
        for key in lista_keys:
            # Extraemos los datos.
            with h5py.File(PATH_DATASET,"r") as file: array = file[key][:]
            # Normalizamos
            vmin,vmax = DIC_NORM[key]
            array = (array - vmin)/(vmax-vmin)
            dic_norm[key] = (vmin,vmax)
            # Ajustamos el shape
            if key == "Altura":
                array = np.reshape(array,(1,*array.shape))
                array = np.ones((num_archivos,num_ubicaciones,resolucion,resolucion))*array
                array = np.reshape(array,(num_datos,resolucion,resolucion))
            else:
                array = np.reshape(array,(num_datos,resolucion,resolucion))
            # Randomizamos
            array = array[idx_rand]
            array_train = array[idx_sep:]
            array_test  = array[:idx_sep]
            array = None # Liberamos memoria de la variable array.
            # Guardamos.
            _shape = array_train.shape[1:]
            chunk_size = (BATCH_SIZE,*_shape)
            
            # Creando el dataset optimizado para la secuenciación en el entrenamiento.
            with h5py.File(PATH_DATASET_TRAIN,"a") as file:
                file.create_dataset(data=array_train,name=key,chunks=chunk_size,dtype=np.float32,compression="gzip")
            
            with h5py.File(PATH_DATASET_TEST,"a") as file:
                file.create_dataset(data=array_test,name=key,chunks=chunk_size,dtype=np.float32,compression="gzip")
            
            print(f"Dataset '{key}' creado con el tamaño: {array_train.shape} , {array_test.shape}")

            with open(PATH_DIC_NORM,"wb") as file:
                pickle.dump(dic_norm,file)

        print(f"\nDiccionario de normalización guardado en:\n{PATH_DIC_NORM}")
        print(f"Datasets creados satisfactoriamente:\n{PATH_DATASET_TRAIN}\n{PATH_DATASET_TEST}")

class SecuenciaHDF5(utils.Sequence):
    """
    Ayuda a iterar sobre un archivo hdf5.
    Los datos en el dataset ya deben de estar normalizados
    y randomizados ya que el iterador solo se encarga de
    suministrar los datos a el bucle de entrenamiento.
    """
    def __init__(self,X_keys,Y_keys,batch_size,path_dataset,callback_y=None,callback_x=None,callback_sample_weight=None):
        
        self.batch_size   = batch_size
        self.path_dataset = path_dataset
        self.X_keys = X_keys
        self.Y_keys = Y_keys
        
        self.callback_x = callback_x
        self.callback_y = callback_y
        self.callback_sample_weight = callback_sample_weight
        
        # Obtenemos el número de datos.
        with h5py.File(path_dataset,"r") as file: 
            self.num_datos = file[X_keys[0]].shape[0]
    
    def __len__(self):
        return self.num_datos // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        X = []
        with h5py.File(self.path_dataset,"r") as file:
            # Preparamos los datos en X
            for key in self.X_keys:
                x = file[key][i : i + self.batch_size]
                X.append(x)
            X = np.stack(X,axis=3)
            
            # Aplicamos la función de callback para X
            if self.callback_x is not None: X = self.callback_x(X)
            
            if self.Y_keys is not None:
                Y = []
                # Preparamos los datos en Y
                for key in self.Y_keys:
                    y = file[key][i : i + self.batch_size]
                    Y.append(y)
                Y = np.stack(Y,axis=3)
            else:
                Y = None
                
            # Aplicamos la función de callback para y
            if self.callback_y is not None: Y = self.callback_y(Y)

        if self.callback_sample_weight is not None:
            return self.callback_sample_weight(X,Y)
        else:
            return X,Y



def preparar_dataset_GAN():
    """
    Para generar el dataset que necesitamos para la GAN, solo hace falta
    que obtener el Cloud Top Height.
    """

    # Revisamos que ya exitan los archivos.
    if os.path.isfile(PATH_DATASET_GAN):
        print("Ya existen los datasets preparados,si desea volver a generarlos eliminelos y vulva a ejecutar la función.") 
    else:
        Keys_input = ["4","6","14","16","Altura","CM"]

        # Revisamos que esté generada el diccionario de normalización.
        print("Cargando diccionario de normalización...")
        with open(PATH_DIC_NORM,"rb") as file:
            dic_norm = pickle.load(file)
        
        # Cargamos los modelos y los pesos de los modelos.
        print("Cargando modelo...")
        modelo_CTH = modelos.Modelo_CTH()
        modelo_CTH.load_weights(f"{DIR_MODELOS}/Pesos/CTH/pesos.tf")
        
        # Caracterizamos el dataset.
        print("Caracterizando el dataset a transformar")
        with h5py.File(PATH_DATASET,"r") as file:
            lista_keys   = list(file.keys())
            num_archivos = file[lista_keys[0]].shape[0]
            num_ubicaciones = file[lista_keys[0]].shape[1]
            resolucion      = file[lista_keys[0]].shape[2]

        # Preparamos randomización de los datos.
        num_datos = num_archivos*num_ubicaciones
        idx_rand = np.arange(num_datos)
        np.random.shuffle(idx_rand)

        # Optimizacón para extraer los datos dado el batch size.
        _shape = (resolucion,resolucion)
        chunk_size = (BATCH_SIZE,*_shape)
        print(f"debug: chunk size {chunk_size}") # ????????????????????????????????
        # GUARDAMOS LOS DATOS DEL CLOUD MASK Y COD
        with h5py.File(PATH_DATASET,"r") as file:
            cm  = file["CM"][:]
            cod = file["COD"][:]

        cm  = cm.reshape((num_datos,resolucion,resolucion))
        cod = cod.reshape((num_datos,resolucion,resolucion)) 

        # Randomizamos
        cm  = cm[idx_rand]
        cod = cod[idx_rand]

        # Normalizamos
        vmin,vmax = dic_norm["COD"]
        cod = (cod-vmin)/(vmax-vmin)

        # Guardamos
        with h5py.File(PATH_DATASET_GAN,"a") as file:
            file.create_dataset(data=cm ,name="CM" ,chunks=chunk_size,dtype=np.float32,compression="gzip")
            file.create_dataset(data=cod,name="COD",chunks=chunk_size,dtype=np.float32,compression="gzip")

        # Liberamos memoria
        cod = None
        cm  = None

        print("Inciando inferencia de los archivos...")
        # OBTENEMOS LOS VALORES DEL CLOUD TOP HEIGHT.
        cth = []
        # Iteramos sobre cada uno de los archivos.
        for index_archivo in range(num_archivos):
            # Extraemos los datos
            with h5py.File(PATH_DATASET,"r") as file:
                # Formalos los vectores de entrada con los datos del batch.
                X = []
                for key in Keys_input:
                    # Ajustamos el shape.
                    if key == "Altura":
                        array = file[key][:]
                    else:
                        array = file[key][index_archivo]
                    # Normalizamos los datos con el diccionario de normalización.
                    vmin , vmax = dic_norm[key]
                    array = (array - vmin) / (vmax - vmin)
                    # Unimos los datos
                    X.append(array) # Shape de los datos (num_loc,resolucion,resolucion)    
                # Formamos el vector de entrada
                X = np.stack(X,axis=3)
                # Hacemos inferencia con el modelo.
                array = modelo_CTH(X).numpy().reshape((num_ubicaciones,resolucion,resolucion))
                cth.append(array)
                print(f"Progreso {index_archivo} de {num_archivos}")
        
        # Unimos todos los datos inferenciados.
        cth = np.stack(cth,axis=0) # Shape (num_archivos,num_loc,res,res)
        cth = cth.reshape((num_datos,resolucion,resolucion))
        cth = cth[idx_rand]
        # Guardando el Cloud Top Height
        with h5py.File(PATH_DATASET_GAN,"a") as file:
            file.create_dataset(data=cth,name="CTH",chunks=chunk_size,dtype=np.float32,compression="gzip")
        
    
            



            






    # Normalizamos los datos usando el diccionario de normalización.



    # Vamos generando el CBM,COD y CTH

    # Y los vamos uniendo con los demas datos.

    # Generamos un dataset con los datos generado. 

    