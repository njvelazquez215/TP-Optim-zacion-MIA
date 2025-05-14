import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn
from jax import debug

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import get_batches, loss, batched_predict


def experimento(data, update, nombreOptimizador, step_size, bs, aux=None, plotFlag=True):
    xx, ff, nx, ny = data

    # Parameters
    num_epochs = 10
    params = init_network_params(layer_sizes, random.key(0))
    params = pack_params(params)

    # initialize gradients
    xi, yi = next(get_batches(xx, ff, bs))
    grads = grad(loss)(params, xi, yi)
    
    if nombreOptimizador == 'RMSProp':
        aux = aux = jnp.square(grads)

    log_train = []
    for epoch in range(num_epochs):
        # Update on each batch
        idxs = random.permutation(random.key(0), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs):
            params, aux = update(params, xi, yi, step_size, aux)

        train_loss = loss(params, xx, ff)
        log_train.append(train_loss)
        if jnp.isnan(train_loss):
            break
        #print(f"Epoch {epoch}, Loss: {train_loss}")
    # Plot loss function
    if plotFlag:
        plt.figure()
        plt.semilogy(log_train)
        plt.title(nombreOptimizador)

        plt.figure()
        plt.imshow(batched_predict(params, xx).reshape((nx, ny)).T, origin='lower', cmap='jet')
        plt.title(nombreOptimizador)

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
    data = datainit(True)

    # ## RMSProp
    nombreOptimizador = 'RMSProp'
    update = update_rmsprop
    step_size = 0.001
    bs = 32
    experimento(data, update, nombreOptimizador, step_size, bs, plotFlag=True)

    # ## SGD
    nombreOptimizador = 'SGD'
    update = update_sgd
    step_size = 0.055
    bs = 33
    experimento(data, update, nombreOptimizador, step_size, bs, plotFlag=True)
    
    

    ## ADAM
    nombreOptimizador = 'ADAM'
    update = update_adam
    k = 0
    beta1 = 0.93
    beta2 = 0.997
    sk = 0
    rk = 0
    aux = (k, beta1, beta2, sk, rk)
    step_size = 0.015
    bs = 40
    experimento(data, update, nombreOptimizador, step_size, bs, aux, plotFlag=True)

    plt.show()



    
    

    
