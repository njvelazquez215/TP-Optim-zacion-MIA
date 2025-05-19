import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from scipy.stats import qmc
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import copy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PyALE import ale
import numpy as np


from implementacion import experimento, datainit
from graficar_resultados import leer_resultado
from nn_functions import update_rmsprop, update_sgd, update_adam


def busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags=None, sslogOverride=True):
    '''
    - Las entradas data, optimizador, schedule y num_epochs son como las entradas de experimento().
    - optimAux y schAux son casi como las de experimento, nada más que cuando se quiere incorporar una de sus variables en el muestreo, en vez de poner
    en un elemento de la tupla un valor numérico, debe ponerse None en su lugar. Si optimAux y/o schAux no son tuplas y son None, directamente este es el valor que se la pasa a experimento, no se muestrea.
    - Si bs es un entero, se realiza la exploración con ese batch size; si es None, se incorpora batch size a la exploración. Las muestras generadas para el batch size se convierten en enteros.
    - Si ss es un escalar se realiza la exploración con ese step size; si es None, se incorpora step size a la exploración.
    - nMuestras es el número de muestras a generar de dimension nParams, en total.
    - rangosParas = [[limInferior1, limSuperior1], [limInferior2, limSuperior2], ...] es una lista con los valores inferiores y superiores de los rangos en los que las muestras pueden tomar valores.
    El orden en el que se generan las muestras son (omitir si no se incluyen): bs, ss, optimAux, schAux. Es importante esto para saber en qué orden dar los rangos.
    '''
    
    nParams = 0

    if bs is None:
        nParams += 1

    if ss is None:
        nParams += 1

    if optimAux is None:
        optimAux = []
    for i in range(len(optimAux)):
        if optimAux[i] is None:
            nParams += 1

    if schAux is None:
        schAux = []
    for i in range(len(schAux)):
        if schAux[i] is None:
            nParams += 1
    
    sampler = qmc.LatinHypercube(d=nParams)  # Generador de muestras LHS
    sample = sampler.random(n=nMuestras)    # Muestras generadas.

    # Para muestreos logarítmicos
    if logscaleFlags is None:
        logscaleFlags = [False for i in nParams]

    if sslogOverride:
            if bs is None:
                logscaleFlags[1] = True
            else:
                logscaleFlags[0] = True
    
    if len(logscaleFlags) != nParams:
        raise(Exception('Número de booleanos para los flag de escalas logarítmicas no coincide con el número de parámetros a muestrear.'))
                
    if any(logscaleFlags):
        for j in range(len(logscaleFlags)):
            if logscaleFlags[j]:
                rangosParams[j] = [jnp.log10(extremo) for extremo in rangosParams[j]]

    # Obtengo la muestra
    sample = qmc.scale(sample, l_bounds=[r[0] for r in rangosParams],  # Se escalan las muestras a los rangos deseados.
                                    u_bounds=[r[1] for r in rangosParams])
    
    # Para muestreos logaritmicos
    if any(logscaleFlags):
        for j in range(len(logscaleFlags)):
            if logscaleFlags[j]:
                sample[:,j] = 10 ** sample[:,j]

    hparams = []

    for i in range(len(sample)):
        hparams.append([])
        
        j = 0   # Contador para ver qué elemento tomar de la muestra

        # Batch size
        if bs is None:
            hparams[i].append(int(jnp.round(sample[i][0])))    # Se transforman los batch sizes a enteros.
            j += 1
        else:
            hparams[i].append(bs)

        # Step size
        if ss is None:
            hparams[i].append(float(sample[i][j]))
            j += 1
        else:
            hparams[i].append(ss)

        # optimAux
        optimAuxList = []
        for k in range(len(optimAux)):
            if optimAux[k] is None:
                optimAuxList.append(float(sample[i][j]))
                j += 1
            else:
                optimAuxList.append(optimAux[k])
        hparams[i].append(tuple(optimAuxList))

        # schAux
        schAuxList = []
        for k in range(len(schAux)):
            if schAux[k] is None:
                schAuxList.append(float(sample[i][j]))
                j += 1
            else:
                schAuxList.append(schAux[k])
        hparams[i].append(tuple(schAuxList))
        
    FF = []     # Inicializo la lista de los costos

    for hparam in tqdm(hparams):    # Se realiza el experimento por cada punto de hiperparámetros
        bs, ss, optimAux, schAux = hparam
        f = experimento(data, optimizador, schedule, num_epochs, ss, bs, optimAux=optimAux, schAux=schAux, plotFlag=False)
        FF.append(f)

    FF = jnp.array(FF)
    FF = jnp.where(jnp.isnan(FF), jnp.inf, FF)  # Para evitar considerar los NaN como mínimos en argmin

    min_idx = jnp.argmin(FF)    # Se obtiene el índice del costo mínimo obtenido.

    # Hiperparámetros correspondientes al mínimo costo
    
    bs, ss, optimAux, schAux = hparams[min_idx]

    resultado = f'El mínimo valor de la función pérdida ({FF[min_idx]}) se obtuvo  con batch size = {bs}, step size = {ss}, optimAux = {optimAux} y schAux = {schAux}.'

    print(resultado)
    # Para el registro
    rec = [hparam[:2] + list(hparam[2]) + list(hparam[3]) + [FF[i]] for i, hparam in enumerate(hparams)]

    fecha = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    descripcion = ['Optimizador: ' + optimizador, 'Schedule: ' + schedule, 'Número de épocas: ' + str(num_epochs), 
                   'Fecha y horario de guardado: ' + fecha, resultado, '', 'Rangos de los hiperparámetros:']
    
    headerRangos = ['límite inferior', 'límite superior']

    header = ['batch size', 'step size']
    for i in range(len(optimAux)):
        header.append('optimAux'+str(i))
    for i in range(len(schAux)):
        header.append('schAux'+str(i))
    
    header.append('loss')

    directorio = 'Resultados/Hiperparámetros'

    os.makedirs(directorio, exist_ok=True)

    nombreArchivo = optimizador + '-' + schedule + '-' + str(num_epochs) + '-' + fecha + '.txt'

    ruta = os.path.join(directorio, nombreArchivo)

    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        for linea in descripcion:
            f.write(linea + '\n')
        writer = csv.writer(f)
        writer.writerow(headerRangos)
        writer.writerows(rangosParams)
        f.write('\nResultados:\n')
        writer.writerow(header)
        writer.writerows(rec)

    return  bs, ss, optimAux, schAux, hparams, min_idx

def estudioALE(rutas):
    if isinstance(rutas, str):
        headers, datos = leer_resultado(rutas)
    elif isinstance(rutas, list):
        headers, datos = leer_resultado(rutas[0])
        for ruta in rutas[1:]:
            np.vstack((datos, leer_resultado(ruta)[1]))
    
    data = {headers[j] : [datos[i,j] for i in range(len(datos[:,j]))] for j in range(len(headers))}
    df = pd.DataFrame(data)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    X = df.drop(columns='loss')
    y = df['loss']
    model = RandomForestRegressor()
    model.fit(X, y)
    print("Train R²:", model.score(X, y))

    wrapped_model = ModelWrapper(model)

    for feature in X.columns:
        print(f'ALE para {feature}')
        ale(X=X, model=wrapped_model, feature=[feature], include_CI=True, plot=True)
        
    plt.show()
    return

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X_input):
        if not hasattr(X_input, "columns"):
            # Reconstruye un DataFrame con los nombres de columna originales
            X_input = pd.DataFrame(X_input, columns=self.model.feature_names_in_)
        return self.model.predict(X_input)

if __name__ == '__main__':
    # Importación y procesamiento de los datos.
    data = datainit(False)

    # Configuración general
    num_epochs = 2

    ## SGD
    optimizador = 'SGD'
    schedule = 'Fix'
    
    bs = None
    ss = None

    optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
    schAux = None       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

    nMuestras = 10
    rangosParams = [[2, 40],        # batch size
                    [0.04, 0.06]]     # step size
    
    #bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams)
    
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True)

    ## ADAM
    optimizador = 'ADAM'
    schedule = 'Power'

    bs = None#30
    ss = 0.02

    beta1 = 0.93
    beta2 = None#0.997

    optimAux = (beta1, beta2)

    r = 50000
    c = None

    schAux = (r, c)

    nMuestras = 2

    # Los rangos de abajo los dejo porque mas o menos funcionaban.
    # rangosParams = [[10, 100],      # batch size
    #                 [0.001, 0.1],   # step size
    #                 [0.7, 0.95],    # beta1
    #                 [0.9, 0.9999]]  # beta2

    rangosParams = [[10, 100],      # batch size
                    [0.9, 0.9999],  # beta2       
                    [0.9, 1]]       # c   
    
    #bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams)
    
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True)

    estudioALE('Resultados/Hiperparámetros/SGD-Fix-50-2025-05-18_18-09-14.txt')
    
    plt.show()