"""
Descripción:
Módulo que contiene lo necesario para el
cálculo del Clearsky.

Nota, solo está adapatado para  trabajar con datos 
del GOES con resolución espacial de 2km

Última modificación:
17 de Mayo del 2022

Autores/Fuentes:
Adrían Ramírez, Facultad de Ciencias, UNAM
felos@ciencias.unam.mx
FelosRG@github

En este módulo se ocupa ampliamente la librería
de código abierto pvlib:
https://pvlib-python.readthedocs.io/en/stable/
"""

import os
import sys
_path_base    = os.path.realpath(__file__) 
_path_base    = "/".join(_path_base.split("/")[:-2])


import h5py
import datetime
import numpy as np

from scipy import ndimage
from pvlib.spa import solar_position , calculate_deltat
from pvlib.irradiance import get_extra_radiation
from pvlib.atmosphere import alt2pres , get_absolute_airmass , get_relative_airmass
from pvlib.clearsky   import ineichen 
from scipy.ndimage import zoom

import GOES

def mapaDeAltura(lat,lon,ventana,nc=2):
    path = _path_base + "/Recursos/CONUS/Altitud_CONUS_2km.h5"
    with h5py.File(path,"r") as dataset:
        array = dataset["Altura"][()]
    x , y = GOES.coordenadas2px(nc,lat,lon)
    array = GOES.obtenerVentana(array,x,y,ventana=ventana)
    return array

def mapaDeTurbidez(lat,lon,ventana,mes,nc=2):
    path = _path_base + "/Recursos/Legacy/LinkeTurbidities_G16_CONUS_375_625.h5"
    with h5py.File(path,"r") as dataset:
        array = dataset["LinkeTurbidity"][()]
    array = array[:,:,mes]
    array = zoom(array,4,order=0,mode="nearest")
    x , y = GOES.coordenadas2px(nc,lat,lon)
    array = GOES.obtenerVentana(array,x,y,ventana=ventana)
    return array
 
def mapaLatLon(lat,lon,ventana,nc=2):
    path = _path_base + "/Recursos/CONUS/Lat_Lon_CONUS_2km.h5"
    x , y = GOES.coordenadas2px(nc,lat,lon)
    with h5py.File(path,"r") as dataset:
        array = dataset["lats"][()]
    lat = GOES.obtenerVentana(array,x,y,ventana=ventana)
    with h5py.File(path,"r") as dataset:
        array = dataset["lons"][()]
    lon = GOES.obtenerVentana(array,x,y,ventana=ventana)
    return lat,lon
    

def _get_solarposition_array(
    time     ,
    latitude ,
    longitude,
    altitude ,
    temperature=None ,
):  
    """
    Adapatación de la función 'get_solarposition' en 
    https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/solarposition.html#get_solarposition

    Parámetros
    -----------

    time: datetime object (UTC)
    latitud : numpy array (float)
    longitud: numpy array (lfloat)

    altitud : numpy array (float)
    temperature: [opcional] numpy array (float)

    Nota:
    De no especificar los parámetros opcionales les serán asignados
    un array con el tamaño del array de latitud con los siguientes valores
    correspondientes:

    temperature : 12

    Que corresponden a valores promedio anuales.
    """

    output_shape  = latitude.shape
    pressure      = alt2pres(altitude=altitude) / 100 # Pasamos a milibares
    atmos_refract = 0.5667
    delta_t       = calculate_deltat(time.year,time.month)

    unixtime = (time - datetime.datetime(1970,1,1)).total_seconds()

    if temperature is None: 
        temperature = np.ones(shape=output_shape)*12.0

    app_zenith , zenith, app_elevation , elevation , azimuth , eot = solar_position(
        unixtime = unixtime  ,
        lat      = latitude  ,
        lon      = longitude ,
        elev     = altitude  ,
        pressure = pressure  ,
        temp     = temperature,
        delta_t  = delta_t    ,
        atmos_refract = atmos_refract ,
    )

    output = {
        "app zenith": app_zenith ,
        "zenith"    : zenith ,
        "app elevation" : app_elevation ,
        "elevation": elevation ,
        "azimuth"  : azimuth   ,
        "eot" : eot ,
    } 

    return output

def mapaZenithAngle(
    time,
    lat,
    lon,
    ventana,
    nc=2,
):
    altura  = mapaDeAltura(lat,lon,ventana,nc)
    lat,lon = mapaLatLon(lat,lon,ventana,nc)
    angulos = _get_solarposition_array(time,lat,lon,altura)

    return angulos["app zenith"]



def mapaClearSky(
    time       ,
    latitud    ,
    longitud   ,
    ventana    ,
    nc = 2     ,
):  

    mes = time.month
    day_of_year = time.timetuple().tm_yday

    # Obtenemos la irradiación extraterrestre
    Io = get_extra_radiation(day_of_year)

    lat, lon = mapaLatLon(latitud,longitud,ventana,nc)
    alt      = mapaDeAltura(latitud,longitud,ventana,nc)
    tur      = mapaDeTurbidez(latitud,longitud,ventana,mes,nc)

    # Obtenemos la posición del sol
    pos_sol = _get_solarposition_array(
        time      = time      ,
        latitude  = lat ,
        longitude = lon ,
        altitude  = alt ,
    )

    # Relative air mass
    relative_am = get_relative_airmass(pos_sol["zenith"])
    
    # Absolute air mass
    absolute_am = get_absolute_airmass(relative_am)

    clearsky = ineichen(
        apparent_zenith  = pos_sol["app zenith"],
        airmass_absolute = absolute_am ,
        linke_turbidity  = tur   ,
        altitude  = alt ,
        dni_extra = Io  ,
    )

    return clearsky


def puntoZenithAngle(time,lat,lon,nc=2,):
    """
    Adaptación rápida del cálculo de zenith para un solo punto.
    Una mejor alternativa sería usar directamente la libería de Pvlib
    """
    ventana=1
    zenit = mapaZenithAngle(time,lat,lon,ventana,nc)
    zenit = np.mean(zenit)
    return zenit

def puntoClearSky(time,latitud,longitud,nc=2):
    """
    Adaptación rápida del cálculo de zenith para un solo punto.
    Una mejor alternativa sería usar directamente la libería de Pvlib
    """
    ventana = 1
    clearsky = mapaClearSky(time,latitud,longitud,ventana,nc)
    for key in clearsky.keys():
        clearsky[key] = np.mean(clearsky[key])
    return clearsky

if __name__ == "__main__":

    print("Test de funcionamiento:")
    print("-----------------------\n")

    time = datetime.datetime.utcnow()
    lat = np.ones((10,10))*20
    lon = np.ones((10,10))*(-100)
    elv = np.ones((10,10))*2000
    pressure = np.ones((10,10))*1000
    temp = np.ones((10,10))*21
    #print(solar_position_numpy(unixtime=unix_time,lat=lat,lon=lon,elev=elv,pressure=pressure,temp=temp,delta_t=10,atmos_refract=1))
    print(_get_solarposition_array(time=time,latitude=lat,longitude=lon,altitude=elv))
    print("\n test completado exitosamente! ")

