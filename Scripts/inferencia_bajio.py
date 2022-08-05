import os
import sys

# Desactivamos mensajes de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Directorios básicos
DIR_SCRIPT  = "/".join(os.path.realpath(__file__).split("/")[:-1])
DIR_REPO    = "/".join(DIR_SCRIPT.split("/")[:-1])
DIR_LIB     = f"{DIR_REPO}/lib"
DIR_MODELOS = f"{DIR_REPO}/Modelos"
DIR_PESOS   = f"{DIR_REPO}/Modelos/Pesos"

# Paths de los recursos
PATH_SHP = f"{DIR_REPO}/Recursos/Shapefiles/shape_file.shp"
PATH_ALT = f"{DIR_REPO}/Recursos/CONUS/Altitud_CONUS_2km.h5"
PATH_CRD = f"{DIR_REPO}/Recursos/CONUS/Lat_Lon_CONUS_2km.h5"

# Agregamos el directorio de modulos al path. 
sys.path.append(DIR_LIB)
sys.path.append(DIR_MODELOS)

import GOES
import modelos

# Librerías en general
import pickle
import h5py
import geopandas as geopd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

def num_ventanas_axis(shape_axis,ventana,solapamiento):
    a = shape_axis - 2*solapamiento
    b = ventana - 2*solapamiento
    if a%b != 0:
        raise ValueError(f"El tamaño de los axis debe de ser multiplo con {b}, tamaño_axis={shape_axis}")
    return a//b

def stride(array,ventana=37,solapamiento=4):
    shape   = array.shape
    itemsize = array.itemsize
    # Calculamos número de ventanas
    shape_out0 = num_ventanas_axis(shape[0],ventana,solapamiento)
    shape_out1 = num_ventanas_axis(shape[1],ventana,solapamiento)
    shape_out  = (shape_out0,shape_out1,ventana,ventana)
    # Calculamos los strides.
    stride0 = itemsize*(ventana-2*solapamiento)*shape[1]
    stride1 = itemsize*(ventana-2*solapamiento)
    stride2 = itemsize*shape[1]
    stride3 = itemsize*1
    strides_out = (stride0,stride1,stride2,stride3)
    return as_strided(array,shape_out,strides_out),shape

def reshape2batch(array):
    """
    Input window (x_array,y_array,ventana,ventana)
    
    Returna:
        * array en forma de batch
        * shape del array en forma ventana
        * strides del array en forma ventana
    """
    shape = array.shape
    num_batches = shape[0]*shape[1]
    ventana = shape[2]
    array_batch = array.reshape(num_batches,ventana,ventana)
    return array_batch,shape

def reshape2window(array,shape):
    """
    Input batch (num_batch,ventana,ventana)
    """
    return array.reshape(shape)
   
def reverse_stride(array,solapamiento=0):
    """
    Input:
        * shape window
    """
    # Caracterizamos el array de entrada
    shape = array.shape
    x_windows = shape[0]
    y_windows = shape[1]
    ventana   = shape[2] - solapamiento*2 
    
    # Creamos el lienzo en blanco para pintar
    array_output = np.zeros((ventana*x_windows,ventana*y_windows))
    
    # Pintamos
    for i in range(x_windows):
        for j in range(y_windows):
            if solapamiento == 0:
                array_output[i*ventana:(i+1)*ventana,j*ventana:(j+1)*ventana] = array[i,j,:,:]
            else:
                array_output[i*ventana:(i+1)*ventana,j*ventana:(j+1)*ventana] = array[i,j,solapamiento:-solapamiento,solapamiento:-solapamiento]
    return array_output

def descargar_bandas():
    GOES.descargar_actual_GOES("ABI-L1b-RadC",banda=4,output_name=f"{DIR_SCRIPT}/Descargas/04.nc")
    GOES.descargar_actual_GOES("ABI-L1b-RadC",banda=6,output_name=f"{DIR_SCRIPT}/Descargas/06.nc")
    GOES.descargar_actual_GOES("ABI-L1b-RadC",banda=14,output_name=f"{DIR_SCRIPT}/Descargas/14.nc")
    GOES.descargar_actual_GOES("ABI-L1b-RadC",banda=16,output_name=f"{DIR_SCRIPT}/Descargas/16.nc")

def plot_bajio(
    array,
    titulo="Bajio",
    cmap="terrain",
    vmin=None,
    vmax=None,
    cbar=False,
    cbar_titulo="Valor",
    titlo_secundario="Proyecto Deep-GOES",
    show=False,
    save="figura.jpg",
    alpha=0.7,
):  
    if vmin is None: vmin = np.min(array)
    if vmax is None: vmax = np.max(array)

    # Abrirmos recursos
    with h5py.File(PATH_CRD,"r") as file:
        lat = file["lats"][:]
        lon = file["lons"][:]
    lat = lat[1079+SOLAPAMIENTO:1348-SOLAPAMIENTO,406+SOLAPAMIENTO:675-SOLAPAMIENTO]
    lon = lon[1079+SOLAPAMIENTO:1348-SOLAPAMIENTO,406+SOLAPAMIENTO:675-SOLAPAMIENTO]

    # Abrimos geopandas
    geodf = geopd.read_file(PATH_SHP)

    fig ,ax = plt.subplots(1,figsize=(19,19))
    # Ploteamos los valores
    colorplot = ax.pcolor(lon,lat,array,shading="nearest",cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
    # Ploteamos divición pólitica
    geodf.plot(ax=ax,facecolor="none")
    # Ploteamos los nombres de los lugares
    ax.scatter(lon_lug,lat_lug,c="red")
    for i, txt in enumerate(lugares):
        ax.annotate(txt, (lon_lug[i]-0.10, lat_lug[i]+0.03),weight="bold")
    # Limitamos el plot a las coordenadas dadas
    ax.set_xlim(-102.1,-99)
    ax.set_ylim(19.88,22.07)

    if cbar:
        cbar = plt.colorbar(colorplot,fraction=0.0355)
        cbar.set_label(cbar_titulo, fontsize=22)
        cbar.ax.tick_params(labelsize=18)
    
    ax.set_title(titulo,loc="left",weight="bold")
    ax.set_title(titlo_secundario,loc="right",weight="bold")

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save,bbox_inches='tight')
    
    plt.close()

localizaciones = {
    "Querétaro":(-100.40,20.603),
    "San Juan del Rio":(-99.98,20.386),
    "Bernal":(-99.940,20.743),
    "Cadereyta":(-99.816,20.695),
    "Jalpan":(-99.472,21.218),
    "Pinal de Amoles":(-99.627,21.135),
    "San Joaquin":(-99.565,20.916),
    "Amealco":(-100.144,20.187),
    "Huimilpan":(-100.274,20.374),
    
    "León":(-101.66,21.120),
    "Guanajuato":(-101.260,21.021),
    "Irapuato":(-101.33,20.68),
    "Celaya":(-100.819,20.534),
    "Moroleon":(-101.181,20.135),
    "Comonfort":(-100.759,20.718),
    "Dolores Hidalgo":(-100.93,21.159),
    "Salvatierra":(-100.879,20.210),
    "Pénjamo":(-101.722,20.429),
    "San Luis de la Paz":(-100.515,21.298),
    "Coroneo":(-100.366,20.200),
    "Acambaro":(-100.719,20.031),
    
    "Ixmiquilpan":(-99.216,20.487),
    "Tecozautla":(-99.634,20.536),
    "Huichapan":(-99.650,20.375),
    
    "Rio Verde":(-99.998,21.937),
    
    "Lagos de Moreno":(-101.926,21.358),
}

SOLAPAMIENTO = 4

# Extraemos la información del diccionario de lugares
lat_lug = []
lon_lug = []
lugares = []
for key,coord in localizaciones.items():
    lugares.append(key)
    lon_lug.append(coord[0])
    lat_lug.append(coord[1])


def rutina(path_figuras):

    # ------------------------------
    # PASO 1: DESCARGA DE LOS DATOS
    # ------------------------------

    print("Descargando datos más recientes...")
    descargar_bandas()
    print("Descarga completada!")

    # ----------------------------
    # PASO 2: PREPARAMOS LOS DATOS
    # ----------------------------

    print("Preparando los datos para la inferencia.")
    # Abrimos archivos y preparamos.
    lista_archivos  = os.listdir(f"{DIR_SCRIPT}/Descargas/")
    lista_fechas = []
    datos  = {}
    for path_producto in lista_archivos:
        # Abrimos producto.
        producto = GOES.Producto(path=f"{DIR_SCRIPT}/Descargas/{path_producto}")
        # Obtenemos y recortamos el array.
        array = producto.array
        array = np.copy(array[1079:1348,406:675]) # El np.copy sirve para restaurar los strides al valor adecuado.
        array , shape_original = stride(array,ventana=37,solapamiento=SOLAPAMIENTO)
        array , shape_ventana  = reshape2batch(array)
        
        # Mandamos los fill value a 0
        fill_value = producto.fill_value 
        array[array == fill_value] = 0
        # Agregamos la fecha a una lista para revisar que todo sea del mismo momento.
        fecha = producto.datetime
        lista_fechas.append(fecha)
        # Obtenemos la banda
        banda = str(producto.banda)
        # Agregamos a diccionario.
        datos[banda] = array
    
    # Revisamos que los datos sean del mismo momento.
    diferencias  = []
    for fecha1 in lista_fechas:
        for fecha2 in lista_fechas:
            dif = (fecha1 - fecha2).total_seconds()
            diferencias.append(dif)
    diferencias = np.array(diferencias)
    if np.max(diferencias) > 180: raise ValueError("Los archivos satélitales no corresponden al mismo momento, volver a descargar.")

    # Abrimos y añadimos array de altura.
    with h5py.File(PATH_ALT,"r") as file:
        altura = file["Altura"][:]
    altura = np.copy(altura[1079:1348,406:675])
    altura , shape_original = stride(altura,ventana=37,solapamiento=SOLAPAMIENTO)
    altura , shape_ventana  = reshape2batch(altura)
    datos["Altura"] = altura

    # Normalizamos datos
    with open(f"{DIR_PESOS}/norm.dic","rb") as file:
        dic_norm = pickle.load(file)
    for key,value in datos.items():
        vmin,vmax = dic_norm[key]
        datos[key] = (value - vmin) / (vmax - vmin)

    # Formamos vectores de entrada.
    Key_X = ["4","6","14","16","Altura"]
    X = []
    for key in Key_X:
        X.append(datos[key])
    X = np.stack(X,axis=3)

    # ------------------------------
    # PASO 3: Realizamos inferencia
    # ------------------------------

    print("Realizando inferencia...")

    modelo_cbm = modelos.Modelo_CBM()
    modelo_cbm.load_weights(f"{DIR_PESOS}/CBM/pesos.tf").expect_partial()
    modelo_cbm.train = False

    modelo_cth = modelos.Modelo_CTH()
    modelo_cth.load_weights(f"{DIR_PESOS}/CTH/pesos.tf").expect_partial()

    modelo_cod = modelos.Modelo_COD()
    modelo_cod.load_weights(f"{DIR_PESOS}/COD/pesos.tf").expect_partial()

    # Obtenemos Cloud Binary Mask    
    cbm = modelo_cbm(X).numpy()
    cbm = cbm.reshape((*cbm.shape,1))
    # Volvemos a formar los vectores de normalización.
    X = np.concatenate([X,cbm],axis=3)

    # Inferencia de los otros modelos
    cth = modelo_cth(X).numpy()
    cod = modelo_cod(X).numpy()
    X = None # Liberando memoria

    # Volvemos a tamaño original
    cbm = reshape2window(cbm,shape_ventana)
    cbm = reverse_stride(cbm,solapamiento=SOLAPAMIENTO)

    cth = reshape2window(cth,shape_ventana)
    cth = reverse_stride(cth,solapamiento=SOLAPAMIENTO)

    cod = reshape2window(cod,shape_ventana)
    cod = reverse_stride(cod,solapamiento=SOLAPAMIENTO)

    # Desnormalizamos
    vmin,vmax = dic_norm["CM"]
    cbm = cbm*(vmax-vmin) + vmin

    vmin,vmax = dic_norm["CTH"]
    cth = cth*(vmax-vmin) + vmin

    vmin,vmax = dic_norm["COD"]
    cod = cod*(vmax-vmin) + vmin

    # -----------------------
    # PASO 5: Generamos plot
    # -----------------------
    print("Generando el plot...")

    cbm = np.round(cbm)
    cbm[cbm == 0.0] = np.nan

    cth = cth / 1000 # Pasamos de m -> km
    cth = cth*cbm

    cod = cod*cbm

    plot_bajio(
        cbm,
        titulo="Presencia de Nubosidad",
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar=False,
        cbar_titulo="Valor",
        titlo_secundario="Proyecto Deep-GOES",
        show=False,
        save=f"{path_figuras}cbm.jpg",
    )

    plot_bajio(
        cth,
        titulo="Altura de tope de las nubes",
        cmap="nipy_spectral",
        vmin=0,
        vmax=17,
        cbar=True,
        cbar_titulo="Altura [Km]",
        titlo_secundario="Proyecto Deep-GOES",
        show=False,
        save=f"{path_figuras}cth.jpg",
        alpha=0.5,
    )

    plot_bajio(
        cod,
        titulo="Opacidad de las nubes",
        cmap="Purples",
        vmin=0,
        vmax=40,
        cbar=True,
        cbar_titulo="Opacidad",
        titlo_secundario="Proyecto Deep-GOES",
        show=False,
        save=f"{path_figuras}cod.jpg",
        alpha=0.95,
    )

    print("Figuras completadas!")

if __name__ == "__main__":
    rutina(path_figuras="")








