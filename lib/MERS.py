"""
Descripción:
Este módulo contiene métodos para el trabajo con el modelo
de estimación de radiación solar MERS.

Última modificación:
16 de Mayo del 2022

Autores/Fuentes:
Adrían Ramírez, Facultad de Ciencias, UNAM
felos@ciencias.unam.mx
FelosRG@github
"""

import os
_path_script    = os.path.realpath(__file__) 
_path_script    = "/".join(_path_script.split("/")[:-1])
_path_root = os.path.realpath(__file__)
_path_root = "/".join(_path_root.split("/")[:-2])

import GOES
import Clearsky as CS


import h5py
import pickle
import netCDF4
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from numpy.lib.stride_tricks import as_strided
from pathlib import Path


def normalizar(
    array,
    valor_min  = 0   ,
    valor_max  = 0   ,
    frac_tomar = 1   ,
    ):
    """
    Clipea  y  normaliza  los  arrays.
    Devuelve el array normalizado junto 
    con los valores de normlaización.
    """
    # Clipeamos a los valores indicados.
    if valor_max != 0 or valor_min != 0:
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

class BloqueDenso(tf.keras.Model):
    def __init__(self,neuronas,salidas,activación="sigmoid"):
        super().__init__()
        self.Denso1 = Dense(neuronas,activation="relu")
        self.Denso2 = Dense(neuronas,activation="relu")
        self.Denso3 = Dense(salidas,activation=activación)
    def call(self,Input):
        Out = self.Denso1(Input)
        Out = self.Denso2(Out)
        Out = self.Denso3(Out)
        return Out

class BloqueConv1x1(tf.keras.Model):
    def __init__(self,filtros,salidas,activación="relu"):
        super().__init__()
        self.Conv1 = Conv2D(filtros,(1,1),activation='relu'     ,padding='same')
        self.Conv2 = Conv2D(filtros,(1,1),activation='relu'     ,padding='same')
        self.Conv3 = Conv2D(salidas,(1,1),activation=activación ,padding='same')
    def call(self,Input):
        Out = self.Conv1(Input)
        Out = self.Conv2(Out)
        Out = self.Conv3(Out)
        return Out

class PositionalEncoder(tf.keras.Model):
    """
    Obtiene la influencia en la radiación global en
    el centro de la imágen satélital de  cada punto 
    según la hora del día y la latitud.
    """
    def __init__(self,shape,neuronas):
        super().__init__()
        self.shape = shape
        # Creamos el mesh que codificará los pesos.
        i = -shape//2
        f =  shape//2 
        x = tf.range(i,f,dtype=tf.float32)
        y = tf.range(i,f,dtype=tf.float32)
        X , Y = tf.meshgrid(x,y)
        self.X = X
        self.Y = Y
        self.Denso1 = Dense(4,activation="tanh")
        self.Denso2 = BloqueDenso(
            neuronas              ,
            salidas= 4 , #self.shape**2
            activación="tanh"     )
        
    def mapaGaussiano(self,xo,yo,sigma,a):
        num_batch = a.shape[0]
        X = tf.expand_dims(self.X,axis=0) 
        Y = tf.expand_dims(self.Y,axis=0)
        X = self.X - xo + 1.0
        Y = self.Y - yo + 1.0
        c = sigma**2
        x = X**2 / (2*c)
        y = Y**2 / (2*c)
        return a*tf.math.exp(-(x+y))

    def call(self,V):
        p = self.Denso1(V)
        p = tf.reshape(p,(-1,4,1,1)) 
        Out = self.mapaGaussiano(p[:,0]*self.shape, p[:,1]*self.shape , p[:,2], p[:,3])
        Out = tf.reshape(Out,(-1,self.shape,self.shape,1))
        return Out

class posGlobales(tf.keras.Model):
    def __init__(self,shape,neuronas):
        super().__init__()
        self.shape  = shape
        self.Denso1 = Dense(neuronas,activation="tanh")
        self.Denso2 = Dense(self.shape**2,activation="tanh")
    def call(self,Input):
        Out = self.Denso1(Input)
        Out = self.Denso2(Out)
        Out = tf.reshape(Out,(-1,self.shape,self.shape,1))
        return Out


class MERS(tf.keras.Model):
    """
    Modelo de Estimación de Radiación Solar:

    3 Modos de operación:

    - Estimación puntual, solo devuleve lso resultados de un solo lugar.
    - Estimacion en área, estima la radiación en una área cuadrada.
    - Estimación nacional, estima la radiación solar en todo el territorio mexicano.

    Los 3 modos requerirán darles como entrada un diccionario con el path a los 
    archivos nc, de cada uno de los canales.
    """

    def __init__(
        self,
        retornar_todo  = False,
        ):
        print("Inicializando variables...")
        super().__init__()
        self.retornar_todo = retornar_todo
        # Paths Arrays Mexico
        self.path_array_coordenadas_MX = _path_root + "/Recursos/Mexico/Lat_Lon_Mexico_2km.h5"
        self.path_array_altitud_MX     = _path_root + "/Recursos/Mexico/Altitud_Mexico_2km.h5"
        self.path_array_turbidity_MX   = _path_root + "/Recursos/Mexico/LinkeTurbidity_Mexico_2km.h5"
        # Paths Arrays CONUS
        self.path_array_coordenadas_CONUS = _path_root + "/Recursos/CONUS/Lat_Lon_CONUS_2km.h5"
        self.path_array_altitud_CONUS     = _path_root + "/Recursos/CONUS/Altitud_CONUS_2km.h5"
        self.path_array_turbidity_CONUS   = _path_root + "/Recursos/CONUS/LinkeTurbidity_CONUS_2km.h5"
        # Parámetros internos
        # Aviso: Hay que reentrenar si se modifican algunos de estos parámetros
        self.neuronas = 25 
        self.shape    = 31
        self.filtros  = 40
        self.canales  = [7,14,15]

        # Keys de diccionarios
        self.key_X = self.canales
        self.key_V = ["Hour","Month","Clearsky GHI","Clearsky DNI","Clearsky DHI","Solar Zenith Angle"]
        self.key_y = ["GHI","DNI","DHI"]
        print("Inicializando capas...")
        # Inicializamos las capas
        self.pos_local  = PositionalEncoder(self.shape,self.neuronas)
        self.pos_global = posGlobales(self.shape,3)
        self.prop_opticas     = BloqueConv1x1(self.filtros,salidas=1,activación="sigmoid")
        self.regresión_local  = Dense(1,activation="sigmoid")
        self.regresión_global = Dense(1,activation="sigmoid")
        print("cargando checkpoint...")
        self.load_weights(_path_root + "/Recursos/MERS/Modelos/modeloSolar3.tf")
        print("Inicialización completa!")

    def call(self,Input):
        X,V = Input
        clearsky_DNI = V[:,3]
        clearsky_DHI = V[:,4]
        zenith = V[:,5]

        clearsky_DNI = tf.reshape(clearsky_DNI,(-1,))
        clearsky_DHI = tf.reshape(clearsky_DHI,(-1,))
        prop_opticas    = self.prop_opticas(X) 
        mask_pos_local  = self.pos_local(V)
        mask_pos_global = self.pos_global(V)
        
        cod_local  = tf.math.multiply(prop_opticas,mask_pos_local )
        cod_global = tf.math.multiply(prop_opticas,mask_pos_global)
        
        total_cod_local = tf.reduce_sum(cod_local,[1,2])
        total_cod_local = self.regresión_local(total_cod_local)
        total_cod_global = tf.reduce_sum(cod_global,[1,2])
        total_cod_global = self.regresión_global(total_cod_global)
        
        total_cod_local  = tf.reshape(total_cod_local,(-1,))
        total_cod_local  = tf.clip_by_value(total_cod_local,0,1)
        total_cod_global = tf.reshape(total_cod_global,(-1,))
        total_cod_global = tf.clip_by_value(total_cod_global,0,1)
        
        DNI = (1 - total_cod_local )*clearsky_DNI
        DHI = (1 - total_cod_global)*clearsky_DHI
        GHI = DHI + tf.math.cos(zenith)*DNI
        
        rad = tf.stack([GHI,DNI,DHI],axis=1)
        
        if self.retornar_todo:
            output = {
                "Radiación"   :rad ,
                "Pesos local" :mask_pos_local,
                "Pesos global":mask_pos_global,
                "Propiedades ópticas":prop_opticas,
            }
            return output
        else:
            return rad

    def _tomarCentro(self,array):
        """
        Toma el dato central de un array.
        """
        shape = array.shape
        if len(shape) == 2:
            x = shape[0]//2 
            y = shape[1]//2
            return array[x,y]
        elif len(shape) == 3:
            x = shape[1]//2 
            y = shape[2]//2
            return array[:,x,y]
        elif len(shape) == 4:
            x = shape[1]//2 
            y = shape[2]//2
            return array[:,x,y,:]
        else:
            raise IndexError("Solo se aceptan array de 2,3,4 dimenciones.")

    def _tensornificar(self,array):
        return tf.constant(array,dtype=tf.float32)

    def _preparar_datos_puntuales(self,latitud,longitud,dic_paths):
        """
        Prepara los datos para la estimación puntual.
        Formato diccionario entrada:
            {int_canal : path_archivo_nc}
        """

        # Obtenemos las coordenadas del centro de la ventana
        px_x,px_y = GOES.coordenadas2px(nc=2,latitud=latitud,longitud=longitud)
        ventana = self.shape // 2

        # Abirmos los datos satélitales
        dic_datos = {} 
        for canal in self.canales:
            nc    = netCDF4.Dataset(dic_paths[canal]) # Abre los archivos del directorio configurado para cada banda
            array = np.array(nc.variables["Rad"])
            array = GOES.obtenerVentana(array,x=px_x,y=px_y,ventana=ventana)
            dic_datos[canal] = array
            dic_datos["t"] = GOES.obtenerFecha(nc=nc,return_datetime=True)
            nc.close()
        
        # Obtenemos la hora en decimal
        hora = dic_datos["t"].hour + dic_datos["t"].minute/60.0
        dic_datos["Hour"] = hora
        # Obtenemos el mes en decimal
        mes = dic_datos["t"].month + dic_datos["t"].day/31.0
        dic_datos["Month"] = mes

        # Obtenemos los arrays de coordenadas.
        with h5py.File(self.path_array_coordenadas_CONUS,"r") as dataset:
            lats = dataset["lats"][()]
            lats = GOES.obtenerVentana(lats,x=px_x,y=px_y,ventana=ventana)
            lons = dataset["lons"][()]
            lons = GOES.obtenerVentana(lons,x=px_x,y=px_y,ventana=ventana)
        dic_datos["Latitud"]  = lats
        dic_datos["Longitud"] = lons

        # Obtenemos el mapa de LinkeTurbidity
        mes = dic_datos["t"].month
        with h5py.File(self.path_array_turbidity_CONUS,"r") as dataset:
            turb = dataset["LinkeTurbidity"][()][:,:,mes]
            turb = GOES.obtenerVentana(turb,x=px_x,y=px_y,ventana=ventana)
            dic_datos["Tur"] = turb
        
        # Obtención del array de Altitud.
        with h5py.File(self.path_array_altitud_CONUS,"r") as dataset:
            alts = dataset["Altura"][()]
            alts = GOES.obtenerVentana(alts,x=px_x,y=px_y,ventana=ventana)
            dic_datos["Alt"] = alts
        
        # Calculamos el ángulo de Zenith
        array = CS._get_solarposition_array(
                time      = dic_datos["t"]        ,
                latitude  = dic_datos["Latitud"]  ,
                longitude = dic_datos["Longitud"] ,
                altitude  = dic_datos["Alt"]      ,
            )["app zenith"]
        dic_datos["Solar Zenith Angle"] = array

        # Calculamos el clearsky
        day_of_year = dic_datos["t"].timetuple().tm_yday
        Io = CS.get_extra_radiation(day_of_year)
        relative_am = CS.get_relative_airmass(dic_datos["Solar Zenith Angle"])
        absolute_am = CS.get_absolute_airmass(relative_am)
        clearsky = CS.ineichen(
            apparent_zenith  = dic_datos["Solar Zenith Angle"],
            airmass_absolute = absolute_am,
            linke_turbidity  = dic_datos["Tur"],
            altitude  = dic_datos["Alt"],
            dni_extra = Io,
            )
        dic_datos["Clearsky GHI"] = clearsky["ghi"]
        dic_datos["Clearsky DNI"] = clearsky["dni"]
        dic_datos["Clearsky DHI"] = clearsky["dhi"]

        # Abrimos el diccionario de normalización
        with open(_path_root + '/Recursos/MERS/Normalizacion.dic', 'rb') as file:
            dic_norm = pickle.load(file)
        
        # Normalizamos arrays
        X = []
        for key in self.key_X:
            array = dic_datos[key].reshape((-1,self.shape,self.shape))
            array , _ ,_ = normalizar(array,dic_norm[str(key)][0],dic_norm[str(key)][1])
            X.append(array)
        X = np.stack(X,axis=3)

        # Normalizamos datos
        V = []

        for key in self.key_V:
            if key == "Hour":
                array = np.ones(shape=(self.shape,self.shape))*dic_datos["Hour"]
            elif key == "Month":
                array = np.ones(shape=(self.shape,self.shape))*dic_datos["Month"]
            else:
                array = dic_datos[key]

            array = array.reshape((-1,self.shape,self.shape))
            v = self._tomarCentro(array)
            v , _ ,_ = normalizar(v,dic_norm[key][0],dic_norm[key][1])
            v = np.array(v)
            V.append(v)
        V = np.stack(V,axis=1)

        return X , V, dic_datos

    def desnormalizarRadiación(self,y):
        with open(_path_root + '/Recursos/MERS/Normalizacion.dic', 'rb') as file:
            dic_norm = pickle.load(file)

        for key in self.key_y:
            y[key] = ( y[key] + dic_norm[key][0] )*dic_norm[key][1]
        
        return y

    def descargarDatosActuales(self,path_descargas=""):
        """
        Descarga las bandas más recientes.
        Los guarda en una carpeta especial
        dentro de la RAM.
        """
        for canal in self.canales:
            GOES.datosActualesGOES16(
                producto    = "ABI-L1b-RadC",
                banda       = canal,
                output_name = path_descargas + str(canal).zfill(2) + ".nc")

    def estimaciónPuntual(self,latitud,longitud,dic_paths):
        """
        Realiza estimación en un solo punto.
        """
        X,V ,dic_datos= self._preparar_datos_puntuales(latitud,longitud,dic_paths)
        X,V = self._tensornificar(X) , self._tensornificar(V)

        if self.retornar_todo:
            y = self.call((X,V))

        else:
            y = self.call((X,V)).numpy()
            out = {
                "latitud" : latitud   ,
                "longitud": longitud  ,
                "UTC": dic_datos["t"] , 
                "GHI": y[0,0],
                "DNI": y[0,1],
                "DHI": y[0,2],
            }
            out = self.desnormalizarRadiación(out)
        
        return out








