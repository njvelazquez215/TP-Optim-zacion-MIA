from implementacion import experimento, datainit
from busquedaHiperparam import busqueda, estudioALE
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------
data = datainit(False)
num_epochs = 50

# --------------------------------------------------------------------
# BÚSQUEDA DE HIPERPARÁMETROS CON ADAM + SCHEDULE EXPONENCIAL
# --------------------------------------------------------------------
optimizador = 'ADAM'
schedule = 'Exponential'

bs = None            # A explorar
ss = None            # A explorar
optimAux = (None, None)  # β1 y β2 a explorar
schAux = (None,)     # `r` del scheduler exponencial

nMuestras = 200
rangosParams = [
    [2, 200],           # batch size
    [0.0001, 0.1],      # step size
    [0.85, 0.99],       # β1
    [0.9, 0.9999],      # β2
    [10000, 500000]     # r (decaimiento exponencial)
]

logscaleFlags = [False, True, False, False, True]  # usamos log-scale para step size y r

# Ejecutar la búsqueda y guardar resultados
bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(
    data, optimizador, schedule, num_epochs,
    bs, ss, optimAux, schAux,
    nMuestras, rangosParams, logscaleFlags
)

# --------------------------------------------------------------------
# ANÁLISIS ALE DE LOS RESULTADOS OBTENIDOS
# --------------------------------------------------------------------
# (Dejar este bloque comentado hasta que sepamos el nombre exacto del archivo generado)
# estudioALE('Resultados/Hiperparámetros/ADAM-Exponential-50-2025-05-20_HH-MM-SS.txt')

plt.show()