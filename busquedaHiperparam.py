import jax.numpy as jnp
from tqdm import tqdm
from scipy.stats import qmc
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os

from implementacion import experimento, datainit
from nn_functions import update_rmsprop, update_sgd, update_adam

def busqueda(data, optimizador, schedule, num_epochs, aux, nParams, nMuestras, rangosParams):
    '''
    - Las entradas data, optimizador, schedule, num_epochs y aux son como las entradas de experimento().
    - En el caso de aux, si el número de hiperparámetros a modificar son más de 2 (y por ahora debe ser el número total de hiperparámetros del optimizador), los últimos hiperparámetros son reemplazados
    por los generados por el muestreo. Es decir, si para ADAM era aux = aux = (k, sk, rk, beta1, beta2), nParams debe ser 4 (o 2) y beta1 y beta2 serán reemplazados por los valores que genere el muestreo.
    Por lo anterior, los elementos de aux deben ubicar a los hiperparámetros al final de la tupla.
    - nParams es el número de hiperparámetros para los cuales se van a generar muestras para lanzar los experimentos.
    Por ahora se supone que los hiperparámetros a utilizar siempre incluyen al batch size y al step size. Las muestras generadas para el batch size se convierten en enteros.
    Por esto, se asume que el primer hiperparámetro a muestrear es el batch size y el segundo el step size.
    Si se pone nParams > 2, se agregan muestras para más variables y por ahora solo quedan como números flotantes comprendidos en los rangos establecidos.
    - nMuestras es el número de muestras a generar de dimension nParams, en total.
    - rangosParas = [[limInferior1, limSuperior1], [limInferior2, limSuperior2], ...] es una lista con los valores inferiores y superiores de los rangos en los que las muestras pueden tomar valores.
    '''

    if nParams < 2:
        raise Exception('Todavía no se implementó.')

    sampler = qmc.LatinHypercube(d=nParams)  # Generador de muestras LHS
    sample = sampler.random(n=nMuestras)    # Muestras generadas.

    hparams = qmc.scale(sample, l_bounds=[r[0] for r in rangosParams],  # Se escalan las muestras a los rangos deseados.
                                    u_bounds=[r[1] for r in rangosParams])

    for i in range(len(hparams)):
        hparams[i][0] = jnp.round(hparams[i][0]).astype(int)    # Se transforman los batch sizes a enteros.

    FF = []     # Inicializo la lista de los costos

    if not aux is None and nParams > 2:     # Si el optimizador necesita la variable aux, se extraen los primero elementos, que no son hiperparámetros modificados.
        auxAux = aux[0:-(len(aux) - (nParams - 2))]
    else: 
        auxAux = None

    for hparam in tqdm(hparams):    # Se realiza el experimento por cada punto de hiperparámetros
        bs = int(hparam[0].item())
        step_size = float(hparam[1].item())

        if not aux is None:
            aux = auxAux + tuple(hparam[2:])    # Se arma la variable aux con los hiperparámetros, si nParams > 2
        
        f = experimento(data, optimizador, schedule, num_epochs, step_size, bs, aux=aux, plotFlag=False)
        FF.append(f)

    FF = jnp.array(FF)
    FF = jnp.where(jnp.isnan(FF), jnp.inf, FF)  # Para evitar considerar los NaN como mínimos en argmin

    min_idx = jnp.argmin(FF)    # Se obtiene el índice del costo mínimo obtenido.

    # Hiperparámetros correspondientes al mínimo costo
    bs = hparams[min_idx][0].astype(int)
    step_size = hparams[min_idx][1]
    
    if not auxAux is None:
        aux = auxAux + tuple(hparams[min_idx][2:])
        resultado = f'El mínimo valor de la función pérdida ({FF[min_idx]}) se obtuvo con step size = {step_size}, batch size = {bs} y aux = {aux}.'
    else:
        resultado = f'El mínimo valor de la función pérdida ({FF[min_idx]}) se obtuvo con step size = {step_size} y batch size = {bs}.'
    print(resultado)

    # Para el registro
    rec = hparams.tolist()
    for i, hparam in enumerate(rec):
        hparam.append(FF[i])

    fecha = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    descripcion = ['Optimizador: ' + optimizador, 'Schedule: ' + schedule, 'Número de épocas: ' + str(num_epochs), 
                   'Fecha y horario de guardado: ' + fecha, resultado, '', 'Rangos de los hiperparámetros:']
    
    headerRangos = ['límite inferior', 'límite superior']

    header = ['batch size', 'step size']
    if nParams > 2:
        for i in range(nParams-2):
            header.append('aux'+str(i))
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

    return step_size, bs, aux, hparams, min_idx


if __name__ == '__main__':
    # Importación y procesamiento de los datos.
    data = datainit(True)

    # Configuración general
    schedule = 'fix'        # Cuando implementemos alguna schedule se cambiaría.
    num_epochs = 10

    ## SGD
    optimizador = 'SGD'

    aux = None

    nParams = 2
    nMuestras = 2
    rangosParams = [[2, 40],        # batch size
                    [0.04, 0.06]]     # step size
    
    step_size, bs, aux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, aux, nParams, nMuestras, rangosParams)
    
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, aux, plotFlag=True)

    ## ADAM
    optimizador = 'ADAM'
    
    beta1 = 0.93
    beta2 = 0.997

    aux = (beta1, beta2)
    
    nParams = 4
    nMuestras = 2
    rangosParams = [[10, 100],      # batch size
                    [0.001, 0.1],   # step size
                    [0.7, 0.95],    # beta1
                    [0.9, 0.9999]]  # beta2
    
    step_size, bs, aux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, aux, nParams, nMuestras, rangosParams)
    
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, aux, plotFlag=True)
    
    plt.show()