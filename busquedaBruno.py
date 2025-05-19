from implementacion import experimento, datainit
from busquedaHiperparam import busqueda, estudioALE

import matplotlib.pyplot as plt

# Importación y procesamiento de los datos.
data = datainit(False)

# Configuración general
num_epochs = 50

##########################
optimizador = 'SGD'
schedule = 'Fix'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = None       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 200
rangosParams = [[2, 200],        # batch size
                [0.0001, 0.1]]     # step size

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams)
#estudioALE('Resultados/Hiperparámetros/SGD-Fix-50-2025-05-18_18-09-14.txt')

# Prueba en un punto particular
ss = 0.03
bs = 1
#experimento(data, optimizador, schedule, num_epochs, ss, bs, plotFlag=True)


##########################
optimizador = 'SGD'
schedule = 'Exponential'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None,)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 200
rangosParams = [[2, 200],        # batch size
                [0.0001, 0.1],      # step size
                [10000, 500000]]    # r
logscaleFlags = [False, True, False]

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_19-48-47.txt')


##########################
# Ahora con escala logarítmica
optimizador = 'SGD'
schedule = 'Exponential'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None,)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 200
rangosParams = [[2, 200],        # batch size
                [0.0001, 0.1],      # step size
                [10000, 500000]]    # r
logscaleFlags = [False, True, True]     # Ahora r con muestreo logarítmico!!!!

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)

#estudioALE('Resultados\Hiperparámetros\SGD-Exponential-50-2025-0z5-18_21-12-43.txt') # Solo primera corrida
#estudioALE('Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_22-18-15.txt') # Solo segunda corrida
#estudioALE(['Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_21-12-43.txt', 'Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_22-18-15.txt'])    # Con las dos corridas

##########################
# Desplazando intervalos.
optimizador = 'SGD'
schedule = 'Exponential'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None,)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 100
rangosParams = [[1, 10],        # batch size
                [0.05, 0.07],      # step size
                [3e5, 3e6]]    # r
logscaleFlags = [False, True, True]     # Ahora r con muestreo logarítmico!!!!

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)

#estudioALE('Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-19_00-53-43.txt')

##########################
optimizador = 'SGD'
schedule = 'Power'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None, None)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 200
rangosParams = [[2, 50],        # batch size
                [0.007, 0.07],      # step size
                [3e5, 3e6],    # r
                [0.6, 1]]       # c
logscaleFlags = [False, True, True, False]

bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)


plt.show()