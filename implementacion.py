import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn
from jax import debug

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import get_batches, loss, batched_predict
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import sch_exponential, sch_power

def experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux=None, schAux=None, plotFlag=True):
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
    - plotFlag: si es True, se grafica.
    '''
    xx, ff, nx, ny = data

    # Parameters
    params = init_network_params(layer_sizes, random.key(0))
    params = pack_params(params)

    # initialize gradients
    xi, yi = next(get_batches(xx, ff, bs))
    grads = grad(loss)(params, xi, yi)
    
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

    t = 1
    log_train = []
    for epoch in range(num_epochs):
        # Update on each batch
        idxs = random.permutation(random.key(0), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs):
            if not scheduleFun is None:
                step_size = scheduleFun(step_size, schAux, t)
            params, optimAux = update(params, xi, yi, step_size, optimAux, t)
            t += 1    
        train_loss = loss(params, xx, ff)
        log_train.append(train_loss)
        if jnp.isnan(train_loss):
            break
        #print(f"Epoch {epoch}, Loss: {train_loss}")
    # Plot loss function
    if plotFlag:
        titulo = optimizador + ' con schedule ' + schedule
        plt.figure()
        plt.semilogy(log_train)
        plt.title(titulo)

        plt.figure()
        plt.imshow(batched_predict(params, xx).reshape((nx, ny)).T, origin='lower', cmap='jet')
        plt.title(titulo)

    return train_loss

def datainit(plotFlag=True):
    # Load data
    field = jnp.load('field.npy')
    field = field - field.mean()
    field = field / field.std()
    field = jnp.array(field, dtype=jnp.float32)
    nx, ny = field.shape
    xx = jnp.linspace(-1, 1, nx)
    yy = jnp.linspace(-1, 1, ny)

    xx, yy = jnp.meshgrid(xx, yy, indexing='ij')
    xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    ff = field.reshape(-1, 1)
    print(f'Hay {len(ff)} datos')
    data = (xx, ff, nx, ny)
    
    if plotFlag:
        # Grafico de la imagen objetivo
        plt.figure()
        plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')
        
        # Lo de abajo es el grafico de la media de las velocidades. Capaz después se puede probar de hacer algo.
        # u = [jnp.mean(field[:,j]) for j in range(ny)]
    
        # plt.figure()
        # plt.plot(yy, u)

    return data

if __name__=='__main__':
    # Importación y procesamiento de los datos.
    data = datainit(True)

    # Experimentos.
    num_epochs = 100

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
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, plotFlag=True)

    schedule = 'Exponential'
    schAux = (100000,)
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, schAux=schAux, plotFlag=True)
    
    schedule = 'Power'
    schAux = (50000, 1)
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, schAux=schAux, plotFlag=True)

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



    
    

    
