import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn
from jax import debug

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import get_batches, loss, batched_predict


def experimento(data, optimizador, schedule, num_epochs, step_size, bs, aux=None, plotFlag=True):
    '''
    - optimizador y schedule son strings con los nombres del optimizador y schedule a utilizar.
    Por ahora están: RMSProp, SGD y ADAM (optimizadores); y fix (schedule).
    - num_epochs: número de épocas.
    - step_size: longitud de paso inicial.
    - bs: batch size.
    - aux: si el optimizador necesita variables auxiliares, se disponen en tuplas aca. 
    SGD no necesita y por lo tanto puede omitirse aux o ser None.
    RMSProp sí necesita pero se calcula internamente, así que vale lo de SGD.
    ADAM sí necesita y en particular son: (beta1, beta2).
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
        aux = aux = jnp.square(grads)
    elif optimizador == 'SGD':
        update = update_sgd
    elif optimizador == 'ADAM':
        update = update_adam
        aux = (0, 0, 0) + aux   # se agrega k, sk, rk, que son el número de iteración y los valores iniciales de sk y rk, que son nulos.
        
    if schedule == 'fix':
        scheduleFun = None

    log_train = []
    for epoch in range(num_epochs):
        # Update on each batch
        idxs = random.permutation(random.key(0), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs):
            params, aux = update(params, xi, yi, step_size, aux)
            if not scheduleFun is None:
                step_size = scheduleFun(step_size)     # Puede que si se use otra rutina haya que pasarle otros parámetros.

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
    num_epochs = 10

    ## RMSProp
    optimizador = 'RMSProp'
    schedule = 'fix'
    step_size = 0.001
    bs = 32
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, plotFlag=True)

    ## SGD
    optimizador = 'SGD'
    schedule = 'fix'
    step_size = 0.055
    bs = 33
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, plotFlag=True)
    
    ## ADAM
    optimizador = 'ADAM'
    schedule = 'fix'
    beta1 = 0.93
    beta2 = 0.997
    aux = (beta1, beta2)
    step_size = 0.015
    bs = 40
    experimento(data, optimizador, schedule, num_epochs, step_size, bs, aux, plotFlag=True)

    plt.show()



    
    

    
