"""
Descripción:
Módulo  que incorcorpora clases y
métodos para trabajar más facilmente 
con los datasets.

Última modificación:
6 de Junio del 2022

Autores/Fuentes:
Adrían Ramírez, Facultad de Ciencias, UNAM
felos@ciencias.unam.mx
FelosRG@github
"""

import h5py
import math
import numpy as np

def normalizarVariable(array,valor_min=None,valor_max=None,frac_tomar=1):
    """
    Clipea  y  normaliza  los  arrays.
    Devuelve el array normalizado junto 
    con los valores de normlaización.
    """
    
    # Clipeamos a los valores indicados.
    if valor_max is not None and  valor_min is not None:
        array = np.clip(array,valor_min,valor_max)
    else:
        valor_max = np.max(array)
        valor_min = np.min(array)
    
    # Normalizamos
    array = (array - valor_min) / valor_max
    
    # Removemos el execeso
    if frac_tomar != 1:
        max_index = np.floor(array.shape[0] *frac_tomar)
        array = array[:int(max_index)]
    
    # Devolvemos array normalizado y valores de normalización.
    return array,valor_min,valor_max


def normalizarDataset(
    diccionario,
    dic_valores_norm = None,
    sigmas           = None,
    tomar_fraccion   = 1,
    ):
    """
    Toma un diccionario con datos y los normaliza.

    dic_valores_norm:
    Es posible introducir un diccionario con los umbrales pre-impuestos
    ejemplo: dic_valores_norm["Hour"] = (0,24)

    sigmas:
    Calcula los valores de mínimo y máximo tomando como umbral
    un cierto número de sigmas (desviaciones estandar)

    Nota:
    Los valores de dic_valores_norm se interponen a los valores
    de sigma.
    """

    dic_datos = {}
    dic_norm  = {}

    for key in diccionario.keys():
        array = diccionario[key]

        valor_min,valor_max = None,None

        # Calcula los valores mínimos y máximos usando los sigmas
        # de las desviaciones estandar.
        if sigmas is not None:
            std = np.std(array)
            umbral    = std*sigmas
            promedio  = np.mean(array)

            valor_min = promedio - umbral
            valor_max = promedio + umbral

            calc_min = np.min(array)
            calc_max = np.max(array)

            if valor_min < calc_min:
                valor_min = calc_min
            if valor_max > calc_max:
                valor_max = calc_max

        if type(dic_valores_norm) is dict:
            if key in dic_valores_norm:
                valor_min,valor_max = dic_valores_norm[key]
                
                # Revisamos el orden correcto si no invertimos
                if valor_min > valor_max:
                    x = valor_min
                    valor_min = valor_max
                    valor_max = x

        array,valor_min,valor_max = normalizarVariable(array,valor_min=valor_min,valor_max=valor_max,frac_tomar=tomar_fraccion)

        dic_datos[key] = array
        dic_norm[key]  = (valor_min,valor_max)
    
    return dic_datos,dic_norm


def desnormalizar(array,variables_norm):
    v_min = variables_norm[0]
    v_max = variables_norm[1]
    return (array*v_max) + v_min 



def randomizarArray(array,orden=None):

    num_datos = array.shape[0]

    if orden is None:
        orden = np.arange(num_datos)
        np.random.shuffle(orden) # Sucede inplace
    
    array = array[orden]

    return array
    
def randomizarDataset(dataset):

    Keys = list(dataset.keys())
    num_datos = dataset[Keys[0]].shape[0]
    
    # Obtenemos el orden
    orden = np.arange(num_datos)
    np.random.shuffle(orden)
    
    for key in Keys:
        dataset[key] = dataset[key][orden]

    return dataset


def preparar_dataset_h5(
    path_input,
    dir_output = "",
    normalizar   = True ,
    dic_norm     = None ,
    randomizar   = True ,
    saltar       = None ,
    test_frac    = None ,
):  

    """
    Descripción: 
    Función que puede normalizar,randomizar y achicar un dataset
    contenido en un archivo HDF.

    Parámetros:
    path_input (str): Path del archivo.h5 con los datos originales
    dir_output (str): Lugar de guardado del nuevo dataset con los
        datos normalizados y ajustados.
    normalizar (bool): [Default True] Si se va o no a normalizar el dataset.
    dic_norm (dict)  : [Default None] Valores preasignados de normalización.
    randomizar (bool): [Default True] Si se randomizan el orden de los datos.
    saltar (int)     : [Default None] El numero de elementos a saltar.
        Ejemplo: Si saltar=3, solo se tomarán un tercio de los datos.
    train_test (float): [Default None] Separa en dos datasets para train y test,
        segun la fracción tomada 
    """

    # Estandarizamos si hace falta el dir_output
    if len(dir_output) != 0: # Ignoramos cuando el dir_output es el el cwd
        if dir_output[-1] != "/":
            dir_output += dir_output + "/"
    
    # Creamos array de normalización cuando este no sea dado
    if normalizar:
        if type(dic_norm) is not dict:
            dic_norm = {}
    
    # Obtenemos los datos del dataset
    with h5py.File(path_input,"r") as file:
        Keys = list(file.keys())
        num_datos = file[Keys[0]].shape[0]
    
    if randomizar:
        indices = np.arange(num_datos)
        np.random.shuffle(indices)
    
    if saltar is not None:
        num_datos = num_datos // saltar
        
    # Iteramos sobre todos los arrays
    for key in Keys:
        with h5py.File(path_input,"r") as file:
            array = file[key][:]
        
        if randomizar:
            array = array[indices]
        
        if saltar is not None:
            array = array[::saltar]
        
        if normalizar:
            if key in dic_norm.keys():
                v_min,v_max = dic_norm[key][0],dic_norm[key][1]
                array,*valores_norm = normalizarVariable(array,valor_min=v_min,valor_max=v_max)
            else:
                array,*valores_norm = normalizarVariable(array)
            dic_norm[key] = valores_norm

        if test_frac is not None:
            index = math.floor(num_datos*test_frac)
            
            array_train = array[index:]
            array_test  = array[:index]
    
        # Creamos nuevos datasets
        if test_frac is None:
            path_output = dir_output + "dataset.h5"
            with h5py.File(path_output,"a") as file:
                file.create_dataset(name=key,data=array,dtype=np.float32)
        else:
            path_output_train = dir_output + "dataset_train.h5"
            path_output_test  = dir_output + "dataset_test.h5"
            with h5py.File(path_output_train,"a") as file:
                file.create_dataset(name=key,data=array_train,dtype=np.float32)
            with h5py.File(path_output_test,"a") as file:
                file.create_dataset(name=key,data=array_test,dtype=np.float32)
    
    # Retornamos el diccionario de normalización.
    if normalizar:
        return dic_norm
    return  None

def preparar_vectores(
    dic_dataset,
    Key_X,
    Key_y,
    Key_V = None,
):
    """
    Descripción:
    Toma como entrada un diccionario con todos los datos
    y los separará entre los vectores.

    """
    X = []
    for key in Key_X:
        array = dic_dataset[key]
        X.append(array)
    X = np.stack(X,axis=-1)

    Y = []
    for key in Key_y:
        array = dic_dataset[key]
        Y.append(array)
    Y = np.stack(Y,axis=-1)

    if Key_V is not None:
        V = []
        for key in Key_V:
            array = dic_dataset[key]
            V.append(array)
        V = np.stack(V,axis=-1)

    if Key_V is not None:
        return X,V,Y
    else:
        return X,Y