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

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\SGD-Power-50-2025-05-19_11-44-24.txt')

##########################
# Intervalos modificados, r no log.
optimizador = 'SGD'
schedule = 'Power'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None, None)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 200
rangosParams = [[1, 10],        # batch size
                [0.04, 0.1],      # step size
                [3e4, 3e6],    # r
                [0.6, 1]]       # c
logscaleFlags = [False, True, False, False]

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\SGD-Power-50-2025-05-19_13-49-25.txt')

##########################
# Intervalos modificados. ss y r fijos.
optimizador = 'SGD'
schedule = 'Power'

bs = None
ss = 0.05

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (2e6, None)       # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).

nMuestras = 100
rangosParams = [[1, 20],        # batch size
                [0.6, 0.8]]       # c
logscaleFlags = [False, False]

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\SGD-Power-50-2025-05-19_15-20-46.txt')

#########################################
#ADAM
#########################################
##########################
optimizador = 'ADAM'
schedule = 'Power'

bs = None
ss = None

optimAux = (0.9, 0.999)
schAux = (None, None)       

nMuestras = 200
rangosParams = [[1, 100],        # batch size
                [0.001, 0.1],   # step size
                [3e4, 3e6],     # r
                [0.6, 1]]       # c
logscaleFlags = [False, True, True, False]

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#axs = estudioALE('Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_18-23-53.txt', logscaleFlags)

##########################
# Intervalos modificados.
optimizador = 'ADAM'
schedule = 'Power'

bs = None
ss = None

optimAux = (0.9, 0.999)
schAux = (None, None)       

nMuestras = 200
rangosParams = [[1, 200],        # batch size
                [0.001, 0.05],   # step size
                [1e2, 1e5],     # r
                [0.6, 1]]       # c
logscaleFlags = [False, True, True, False]

#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#axs = estudioALE('Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_21-59-15.txt', logscaleFlags)

##########################
# Junto los ale de los resultados anteriores.
#axs = estudioALE(['Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_18-23-53.txt', 'Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_21-59-15.txt'], logscaleFlags)
##########################
# Intervalos modificados. bs y c fijos
optimizador = 'ADAM'
schedule = 'Power'

bs = 120
ss = None

optimAux = (0.9, 0.999)
schAux = (None, 0.75)       

nMuestras = 200
rangosParams = [[0.005, 0.1],   # step size
                [1e4, 5e5]]     # r

logscaleFlags = [True, True]
#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#axs = estudioALE('Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_22-58-05.txt', logscaleFlags)

##########################
# Intervalos modificados. Mas epochs, beta1, beta2 y r
num_epochs = 500
optimizador = 'ADAM'
schedule = 'Power'

bs = 120
ss = 0.03

optimAux = (None, None)
schAux = (None, 0.75)       

nMuestras = 200
rangosParams = [[0.6, 0.95],    # beta 1
                [0.9, 0.9999],  # beta 2
                [1e4, 5e5]]     # r

logscaleFlags = [False, False, True]
#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
#axs = estudioALE('Resultados\Hiperparámetros\ADAM-Power-500-2025-05-20_06-00-10.txt', logscaleFlags)

##########################
num_epochs = 6
optimizador = 'ADAM'
schedule = 'Power'

bs = 120
ss = 0.03

optimAux = (0.9, 0.99)
schAux = (2e4, 0.75)       

epochsHistoHess = [0, 1, 5]
lmdb = None#1e-5
#experimento(data, optimizador, schedule, num_epochs, ss, bs, optimAux, schAux, True, epochsHistoHess=epochsHistoHess, lmbd=lmdb)


##########################
# Performance
nMuestras = 200
num_epochs = 50
optimizador = 'ADAM'
schedule = 'Performance'

bs = None
ss = None

optimAux = (0.9, 0.999)
schAux = (None, None)

rangosParams = [[10, 200],        # batch size
                [0.001, 0.05],   # step size
                [1, 4],          # schAux1
                [0.01, 1]        # schAux2
                ]
logscaleFlags = [False, True, True, False]
#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)

bs = 60
ss = 0.003

optimAux = (None, None)
schAux = (1.05, 0.25)

rangosParams = [[0.4, 0.95],    # beta 1
                [0.9, 0.9999]]  # beta 2

logscaleFlags = [False, False]
#bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)

############################
optimizador = 'SGD'
schedule = 'Performance'

bs = None
ss = None

optimAux = None     # Es None porque no se requiere, no por el muestreo (si este fuera el caso, sería una tupla con algún elemento None).
schAux = (None, None)       

nMuestras = 200
rangosParams = [[1, 100],        # batch size
                [0.001, 0.05],   # step size
                [1, 4],          # schAux1
                [0.01, 1]]        # schAux2
                
logscaleFlags = [False, True, False, True]

bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams, logscaleFlags)
plt.show()