from implementacion import experimento, datainit
from busquedaHiperparam import busqueda, estudioALE
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------
data = datainit(False)
num_epochs = 50

# --------------------------------------------------------------------
# BÚSQUEDA DE HIPERPARÁMETROS CON ADAM (Paso Fijo)
# --------------------------------------------------------------------
optimizador = 'ADAM'
schedule = 'Fix'

bs = None            # A explorar
ss = None            # A explorar
optimAux = (None, None)  # β1 y β2 a explorar
schAux = None        # No se explora (Fix)

nMuestras = 200
rangosParams = [
    [2, 200],           # batch size
    [0.0001, 0.1],      # step size (logscale ideal)
    [0.85, 0.99],       # β1
    [0.9, 0.9999]       # β2
]

logscaleFlags = [False, True, False, False]

# Ejecutar la búsqueda y guardar resultados
bs, step_size, optimAux, schAux, hparams, min_idx = busqueda(
    data, optimizador, schedule, num_epochs,
    bs, ss, optimAux, schAux,
    nMuestras, rangosParams, logscaleFlags
)

# --------------------------------------------------------------------
# ANÁLISIS ALE DE LOS RESULTADOS OBTENIDOS
# --------------------------------------------------------------------
# (Dejar este bloque comentado hasta que se sepa el nombre exacto del archivo generado)
# estudioALE('Resultados/Hiperparámetros/ADAM-Fix-50-2025-05-20_10-57-28.txt')

# Si ya tenemos varios .txt y queremos fusionar resultados:
# estudioALE([
#     'Resultados/Hiperparámetros/ADAM-Fix-50-2025-05-20_13-00-00.txt',
#     'Resultados/Hiperparámetros/ADAM-Fix-50-2025-05-20_14-20-10.txt'
# ])

plt.show()