import jax.numpy as jnp
from jax import grad, jit, vmap, hessian
from jax import random
from jax import nn
from jax import debug
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime
from nn_functions import init_network_params, pack_params, layer_sizes, unpack_params
from nn_functions import get_batches, loss, batched_predict
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import sch_exponential, sch_power, sch_CLR
from nn_functions import batched_activaciones

def experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux=None, schAux=None, plotFlag=True, PSWAAux=None, epochsHistoHess=None, lmbd=None, epochFourier=None, lmbd_grad=None, printFlag=False, guardarLog=False):
    '''
    - optimizador y schedule son strings con los nombres del optimizador y schedule a utilizar.
    Por ahora están: RMSProp, SGD y ADAM (optimizadores); y Fix, Exponential y Power (schedule).
    - num_epochs: número de épocas.
    - step_size: longitud de paso inicial.
    - bs: batch size.
    - optimAux: si el optimizador necesita variables auxiliares, se disponen en tuplas aca. 
    SGD no necesita y por lo tanto puede omitirse optimAux o ser None.
    RMSProp sí necesita pero se calcula internamente, así que vale lo de SGD.
    ADAM sí necesita y en particular son: (beta1, beta2).
    - schAux: en esta tupla se almacenan los hiperparametros necesarios para la schedule utilizada para el learning rate.
    Fix no necesita
    Exponential requiere que sea: (r)
    Power requiere que sea: (r, c)
    CLR requiere que sea: (a1, a2_a1, c) donde a1 es el valor de LR más alto del ciclo, a2_a1 es la relacion de a2 con a1 y c es el número de iteraciones que conlleva cada ciclo.
    - plotFlag: si es True, se grafica.
    PSWAAux: (cSWA, e1SWA) donde cSWA es el número de promedios que se realizan para realizar una actualización de params con el promedio (se realiza un promedio al finalizar una epoch),
    y e1SWA es el epoch en el que se realiza el primer promedio.
    '''
    xx, ff, nx, ny, auxDataMod, grad_ff = data
    if auxDataMod:
        u, fieldModMean, fieldModStd = auxDataMod

    # Parameters
    params = init_network_params(layer_sizes, random.key(0))
    params = pack_params(params)

    # initialize gradients
    xi, yi, gradi = next(get_batches(xx, ff, grad_ff, bs))
    grads = grad(loss)(params, xi, yi, gradi, lmbd, lmbd_grad)
    
    if optimizador == 'RMSProp':
        update = update_rmsprop
        optimAux = optimAux = jnp.square(grads)
    elif optimizador == 'SGD':
        update = update_sgd
    elif optimizador == 'ADAM':
        update = update_adam
        optimAux = (0, 0) + optimAux   # se agrega sk, rk, que son el número de iteración y los valores iniciales de sk y rk, que son nulos.
    
    if schedule == 'Fix':
        scheduleFun = None
    elif schedule == 'Exponential':
        scheduleFun = sch_exponential
    elif schedule == 'Power':
        scheduleFun = sch_power
        schAux = (step_size,) + schAux
    elif schedule == 'CLR':
        scheduleFun = sch_CLR
    elif schedule == 'Performance':
        scheduleFun = None
        if schAux[0] < 1:
            raise ValueError('El primer elemento de schAux debe ser mayor o igual a 1 para la schedule Performance.')
        if schAux[1] > 1:
            raise ValueError('El segundo elemento de schAux debe ser menor o igual a 1 para la schedule Performance.')

    if PSWAAux:
        cSWA, e1SWA = PSWAAux
        nSWA = None
        paramsSWA = 0

    t = 1
    log_train = []
    log_min = None
    for epoch in range(num_epochs):
        if epochsHistoHess and epoch in epochsHistoHess:
            plotActivacionesHess(params, xi, yi, gradi, epoch, lmbd, lmbd_grad)
        if epochFourier and epoch in epochFourier:
            plotFourier(ff, params, xx, nx, ny, epoch)

        # Update on each batch
        idxs = random.permutation(random.key(0), xx.shape[0])
        for xi, yi, gradi in get_batches(xx[idxs], ff[idxs], grad_ff[idxs], bs):
            if not scheduleFun is None:
                step_size = scheduleFun(step_size, schAux, t)
            params, optimAux = update(params, xi, yi, gradi, step_size, optimAux, t, lmbd, lmbd_grad)
            t += 1

        train_loss = loss(params, xx, ff, grad_ff, lmbd=None, lmbd_grad=None)

        if schedule == 'Performance':
            if epoch == 0:
                train_loss_old = train_loss
            else:
                if train_loss < train_loss_old:
                    step_size *= schAux[0]
                else:
                    step_size *= schAux[1]
                train_loss_old = train_loss

        if auxDataMod:
            train_loss = train_loss * fieldModStd ** 2
        log_train.append(train_loss)

        if jnp.isnan(train_loss):
            break

        if log_min is None:
            log_min = (train_loss, params)
        else:
            if train_loss < log_min[0]:
                log_min = (train_loss, params)
        
        # PSWA
        if PSWAAux:
            if nSWA is None:
                if (epoch + 1) % e1SWA == 0:
                    nSWA = 0
            if not nSWA is None:
                paramsSWA = (paramsSWA * nSWA+ params)/(nSWA + 1)
                nSWA += 1
                if nSWA % cSWA == 0:
                    print('SWA')
                    params = paramsSWA
                    nSWA = 0
                    paramsSWA = 0
        if printFlag:
            print(f"Epoch {epoch}, Loss: {train_loss}")
    train_loss, params = log_min
    log_train.append(train_loss)

    #if auxDataMod:
        # preds = batched_predict(params, xx).reshape((nx, ny))
        # preds = preds * fieldModStd + fieldModMean + u
        # preds = preds.reshape(-1, 1)
        # target = ff.reshape((nx,ny)) * fieldModStd + fieldModMean + u
        # target = target.reshape(-1, 1)
        # loss_data = jnp.mean(jnp.square(preds - target))
        # if not lmbd is None:
        #     weights = unpack_params(params)
        #     l2_penalty = sum(jnp.sum(w**2) for w, _ in weights)
        #     loss_data += lmbd * l2_penalty
        # print(f'La pérdida equivalente si no se hubieran preprocesado los datos es: {loss_data}')
       
        # Lo de abajo es equivalente a todo lo anterior!
        #print(f'La pérdida equivalente si no se hubieran preprocesado los datos es: {train_loss * fieldModStd ** 2}')
        

    # Plot loss function
    if plotFlag:
        partes = [optimizador, schedule]

        if lmbd is not None and lmbd != 0:
            partes.append("L2")
        elif lmbd_grad is not None and lmbd_grad != 0:
            partes.append("grad")

        if auxDataMod is not None:
            partes.append("con preprocesamiento")

        titulo = " ".join(partes)

        plt.figure()
        plt.semilogy(log_train)
        plt.title(titulo)

        im = batched_predict(params, xx).reshape((nx, ny))
        if auxDataMod:
            im = im * fieldModStd + fieldModMean + u
        plt.figure()
        plt.imshow(im.T, origin='lower', cmap='jet')
        plt.title(titulo)
    if printFlag: 
        print(f'El valor mínimo de la pérdida es: {train_loss}')

    if guardarLog:
        guardar_log_train(log_train, optimizador, schedule, lmbd, lmbd_grad, auxDataMod)

    return train_loss

def guardar_log_train(log_train, optimizador, schedule, lmbd, lmbd_grad, auxDataMod):
    carpeta="Resultados/log_train"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if lmbd is not None or lmbd == 0:
        reg_suffix = "L2"
    elif lmbd_grad is not None or lmbd_grad == 0:
        reg_suffix = "grad"
    else:
        reg_suffix = "None"

    pre_suffix = "pre" if auxDataMod is not None else "None"

    filename = f"{optimizador}-{schedule}-{reg_suffix}-{pre_suffix}-{timestamp}.txt"
    filepath = carpeta + '/' + filename

    # Guardar log en el archivo
    with open(filepath, 'w') as f:
        for loss in log_train:
            f.write(f"{loss}\n")

def graficar_logs(lista_archivos):
    fig, ax = plt.subplots()

    for file in lista_archivos:
        with open(file, 'r') as f:
            log_train = [float(line.strip()) for line in f.readlines()]

        filename = file.split('/')[-1].split('\\')[-1].replace('.txt', '')
        partes = filename.split('-')
        optim = partes[0]
        schedule = partes[1]
        reg = partes[2]
        pre = partes[3]
        if reg == 'None':
            reg = ''
        if pre == 'None':
            pre = ''
        else:
            pre = 'con preprocesamiento'
        label = f"{optim} {schedule} {reg} {pre}"

        ax.semilogy(log_train, label=label)

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()

def plotFourier(ff, params, xx, nx, ny, epoch):
    ff = ff.reshape(nx, ny)
    predict = batched_predict(params, xx).reshape(nx, ny)

    idx = ny // 2

    ff = ff[:,idx]
    predict = predict[:,idx]
    
    xx = jnp.linspace(-1, 1, nx)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('Corte longitudinal medio de los datos y el ajuste')
    ax2 = fig.add_subplot(122)
    ax2.set_title('Espectro de amplitud del corte longitudinal medio')

    ax1.plot(xx, ff, label='Campo original', color='b')
    ax1.plot(xx, predict, label='Ajuste de NN', color='r')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('u normalizada')
    
    ax1.legend()

    ff = jnp.fft.fft(ff)
    ff = jnp.abs(ff) / nx

    predict = jnp.fft.fft(predict)
    predict = jnp.abs(predict) / nx

    frecuencias = jnp.fft.fftfreq(nx)[: nx // 2]

    ff = ff[: nx // 2]
    predict = predict[: nx // 2]

    ax2.scatter(frecuencias, ff, color='b', label='Campo original', s=10)
    ax2.scatter(frecuencias, predict, color='r', label='Ajuste de NN', s=10)
    # ax2.vlines(frecuencias, ymin=1e-10, ymax=ff, color='b', label='Campo original')
    # ax2.vlines(frecuencias, ymin=1e-10, ymax=predict, color='r', label='Ajuste de NN', alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(f'Época {epoch}')

def plotActivacionesHess(params, xi, yi, gradi, epoch, lmbd, lmbd_grad):
    nBins = 20
    # Histograma de activaciones
    activaciones = batched_activaciones(params, xi)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    for i, activacion in enumerate(activaciones):
        flat = jnp.ravel(activacion)
        ax1.hist(flat, bins=nBins, alpha=0.6, label=f'Capa {i}')
        ax1.legend()
    ax1.set_title('Histograma de activaciones')

    # Histograma del espectro del Hessiano

    H = hessian(loss)
    H = H(params, xi, yi, gradi, lmbd, lmbd_grad)
    eig= jnp.linalg.eigvalsh(H)

    eigPos = eig[eig > 0]
    eigNeg = abs(eig[eig < 0])

    binsPos = np.logspace(np.log10(eigPos.min()), np.log10(eigPos.max()), nBins // 2)
    binsNeg = np.logspace(np.log10(eigNeg.min()), np.log10(eigNeg.max()), nBins // 2)
    binsNeg = - binsNeg[::-1]

    bins = np.concatenate([binsNeg, [0], binsPos])

    ax2 = fig.add_subplot(2, 1, 2)

    ax2.hist(eig, bins=bins, edgecolor='k')
    ax2.set_xscale('symlog', linthresh=1e-12)

    ax2.set_xlabel('Autovalor')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Espectro de la hessiana')
    fig.suptitle(f'Época {epoch}')

    
    
def datainit(plotFlag=True, modFlag=False):
    # Load data
    field = jnp.load('field.npy')
    field = field - field.mean()
    field = field / field.std()
    field = jnp.array(field, dtype=jnp.float32)
    nx, ny = field.shape
    xx = jnp.linspace(-1, 1, nx)
    yy = jnp.linspace(-1, 1, ny)

    xx, yy = jnp.meshgrid(xx, yy, indexing='ij')

    if modFlag:
        u = jnp.mean(field, axis=0)

        fieldMod = field - u[None,:]
        fieldModMean = fieldMod.mean()
        fieldMod = fieldMod - fieldModMean
        fieldModStd = fieldMod.std()
        fieldMod = fieldMod / fieldModStd

        auxDataMod = (u[None,:], fieldModMean, fieldModStd)
    else:
        auxDataMod = None
    
    if plotFlag:
        if modFlag:
            xx_np = np.array(xx)
            yy_np = np.array(yy)
            field_np = np.array(field)
            u_np = np.array(u)
            fieldMod_np = np.array(fieldMod)
            fig = plt.figure()

            ax = fig.add_subplot(131)
            ax.plot(yy_np[0,:], u_np)
            ax.set_title('Distribución de velocidades medias')

            ax = fig.add_subplot(132, projection='3d')
            ax.plot_surface(xx_np, yy_np, field_np)
            ax.set_title('Datos originales')
            
            ax=fig.add_subplot(133, projection='3d')
            ax.plot_surface(xx_np, yy_np, fieldMod_np)
            ax.set_title('Datos procesados')
        
        plt.figure()
        plt.imshow(field.T, origin='lower', cmap='jet')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Imágen objetivo')
    
    xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    
    if modFlag:
        field = fieldMod

    df_dx, df_dy = np.gradient(field, 2 / (nx - 1), 2 / (ny - 1))
    df_dx = jnp.array(df_dx).flatten()
    df_dy = jnp.array(df_dy).flatten()

    grad_ff = jnp.stack([df_dx, df_dy], axis=-1)

    ff = field.reshape(-1, 1)

    print(f'nx = {nx}, ny = {ny}, por lo que hay {len(ff)} datos')

    data = (xx, ff, nx, ny, auxDataMod, grad_ff)
    return data

# def datainitMod(plotFlag=True):
#     field = jnp.load('field.npy')
#     field = field - field.mean()
#     field = field / field.std()
#     field = jnp.array(field, dtype=jnp.float32)
#     nx, ny = field.shape
#     xx = jnp.linspace(-1, 1, nx)
#     yy = jnp.linspace(-1, 1, ny)
#     xx, yy = jnp.meshgrid(xx, yy, indexing='ij')

#     u = jnp.mean(field, axis=0)

#     fieldMod = field - u[None,:]
#     fieldModMean = fieldMod.mean()
#     fieldMod = fieldMod - fieldModMean
#     fieldModStd = fieldMod.std()
#     fieldMod = fieldMod / fieldModStd

#     auxDataMod = (u[None,:], fieldModMean, fieldModStd)

#     if plotFlag:
#         xx_np = np.array(xx)
#         yy_np = np.array(yy)
#         field_np = np.array(field)
#         u_np = np.array(u)
#         fieldMod_np = np.array(fieldMod)
#         fig = plt.figure()

#         ax = fig.add_subplot(131)
#         ax.plot(yy_np[0,:], u_np)

#         ax = fig.add_subplot(132, projection='3d')
#         ax.plot_surface(xx_np, yy_np, field_np)
        
#         ax=fig.add_subplot(133, projection='3d')
#         ax.plot_surface(xx_np, yy_np, fieldMod_np)
    
#     xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
#     ff = fieldMod.reshape(-1, 1)
#     print(f'Hay {len(ff)} datos')
#     data = (xx, ff, nx, ny)
#     return data, auxDataMod
    

if __name__=='__main__':
    # Importación y procesamiento de los datos.
    data = datainit(True)

    # Experimentos.
    num_epochs = 10

    ## RMSProp
    optimizador = 'RMSProp'
    schedule = 'Fix'
    step_size = 0.001
    bs = 32
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, plotFlag=True)

    ## SGD
    optimizador = 'SGD'
    step_size = 0.055
    bs = 33
    
    schedule = 'Fix'
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, plotFlag=True)

    schedule = 'Exponential'
    schAux = (100000,)
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, schAux=schAux, plotFlag=True)
    
    schedule = 'Power'
    schAux = (50000, 1)
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, schAux=schAux, plotFlag=True)

    ## ADAM
    optimizador = 'ADAM'
    
    beta1 = 0.93
    beta2 = 0.997
    optimAux = (beta1, beta2)
    step_size = 0.015
    bs = 40

    schedule = 'Fix'
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, plotFlag=True)

    schedule = 'Exponential'
    schAux = (50000,)
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True)

    schedule = 'Power'
    schAux = (50000, 1)
    #experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=True)

    plt.show()



    
    

    
