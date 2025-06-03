from implementacion import experimento, datainit, graficar_logs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def guardar_loss_reg(reg, losses, tipo):
    """
    Guarda en un archivo los valores finales de loss asociados a cada valor de regularización L2.
    """
    if tipo not in ['L2', 'grad']:
        raise ValueError("Tipo debe ser 'L2' o 'grad'.")
    carpeta="Resultados"
    filepath = f"{carpeta}/loss-{tipo}.txt"
    with open(filepath, 'w') as f:
        for l2, loss in zip(L2, losses):
            f.write(f"{l2}\t{loss}\n")

def graficar_loss_reg(tipo):
    """
    Lee el archivo con valores de regularización L2 y sus correspondientes pérdidas finales, y grafica el resultado.
    """
    if tipo not in ['L2', 'grad']:
        raise ValueError("Tipo debe ser 'L2' o 'grad'.")
    filepath=f"Resultados/loss-{tipo}.txt"
    L2 = []
    losses = []

    with open(filepath, 'r') as f:
        for line in f:
            l2, loss = map(float, line.strip().split())
            L2.append(l2)
            losses.append(loss)

    fig, ax = plt.subplots()
    ax.semilogx(L2, losses)
    ax.set_xlabel(f"Regularización {tipo} (lambda)")
    ax.set_ylabel("Loss final")

# General
#data = datainit(plotFlag=False, modFlag=False)
num_epochs = 1000

optimizador = 'ADAM'

## Power
optimAux = (0.9, 0.99)

schedule = 'Power'
schAux = (30000, 0.75)

bs = 120
step_size = 0.03

#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True, printFlag=True,guardarLog=True)

## Performance
optimAux = (0.6, 0.98)

schedule = 'Performance'
schAux = (1.05, 0.25)

bs = 60
step_size = 0.002

#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True, printFlag=True, guardarLog=True)

#graficar_logs([Resultados\log_train\ADAM-Performance-2025-None-06-02_20-19-02.txt', 'Resultados\log_train\ADAM-Power-None-2025-06-02_20-20-47.txt'])


schAux = (1, 0.25)
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True, printFlag=True, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Performance-2025-None-06-02_20-35-01.txt', 'Resultados\log_train\ADAM-Power-None-2025-06-02_20-20-47.txt'])

################################################
# Esquema preseleccionado
optimAux = (0.9, 0.99)

schedule = 'Power'
schAux = (30000, 0.75)

bs = 120
step_size = 0.03

# Datos preprocesados
#data = datainit(plotFlag=True, modFlag=True)

#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True, printFlag=True, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Power-None-None-2025-06-02_20-20-47.txt','Resultados\log_train\ADAM-Power-None-pre-2025-06-02_20-56-00.txt'])

# Datos preprocesados, mismo número de épocas
num_epochs = 335
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True, printFlag=True, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Power-None-None-2025-06-02_20-20-47.txt','Resultados\log_train\ADAM-Power-None-pre-2025-06-02_20-56-00.txt'])

################################################
data = datainit(plotFlag=False, modFlag=True)

# L2
num_epochs = 100
L2 = np.logspace(-5, 0, 30)
L2 = np.insert(L2, 0, 0.0)
losses=[]
# for l2 in tqdm(L2):
#     losses.append(experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd=l2, printFlag=False, plotFlag=False))
# guardar_loss_reg(L2, losses, tipo='L2')
# graficar_loss_reg(tipo='L2')

# L2 histogramas
num_epochs = 6
lmbd = None
epochHistoHess = [1, 2, 5]
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd=lmbd, printFlag=True, plotFlag=False, epochsHistoHess=epochHistoHess)
lmbd = 1e-3
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd=lmbd, printFlag=True, plotFlag=False, epochsHistoHess=epochHistoHess)
num_epochs = 100
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd=lmbd, printFlag=True, plotFlag=True, guardarLog=True)
lmbd = 0
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd=lmbd, printFlag=True, plotFlag=True, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Power-L2-pre-2025-06-02_23-01-54.txt','Resultados\log_train\ADAM-Power-None-pre-2025-06-02_23-02-05.txt'])

# GRAD
num_epochs = 300
grad = np.logspace(-5, 0, 30)
grad = np.insert(grad, 0, 0.0)
losses=[]
# for gr in tqdm(grad):
#     losses.append(experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd_grad=gr, printFlag=False, plotFlag=False))
# guardar_loss_reg(grad, losses, tipo='grad')
# graficar_loss_reg(tipo='grad')


# Comparación preprocesado sin y con regularización supervisada con el gradiente
num_epochs = 1000
lmbd_grad = 0.00025
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd_grad=lmbd_grad, printFlag=True, plotFlag=True, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Power-None-None-2025-06-02_20-20-47.txt','Resultados\log_train\ADAM-Power-None-pre-2025-06-02_20-56-00.txt','Resultados\log_train\ADAM-Power-grad-pre-2025-06-03_08-44-59.txt'])

# Fourier. Comparacion con gradiente, pero datos con y sin preprocesar
num_epochs = 4000
epochFourier = [1, 10, 100, 500, 1000, 3999]
epochHistoHess = [3999]
experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd_grad=lmbd_grad, printFlag=True, plotFlag=True, epochFourier=epochFourier, guardarLog=True, epochsHistoHess=epochHistoHess)

data = datainit(plotFlag=False, modFlag=False)
#experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, lmbd_grad=lmbd_grad, printFlag=True, plotFlag=True, epochFourier=epochFourier, guardarLog=True)
#graficar_logs(['Resultados\log_train\ADAM-Power-grad-None-2025-06-03_11-01-27.txt', 'Resultados\log_train\ADAM-Power-grad-pre-2025-06-03_10-49-35.txt'])

plt.show()