from implementacion import experimento, datainit
from busquedaHiperparam import busqueda

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

bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(data, optimizador, schedule, num_epochs, bs, ss, optimAux, schAux, nMuestras, rangosParams)

experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True)