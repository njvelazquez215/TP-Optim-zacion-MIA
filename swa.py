# -----------------------------------------------------------
# Punto 3.b del Trabajo Práctico - Stochastic Weight Averaging
# -----------------------------------------------------------

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import matplotlib.pyplot as plt

from nn_functions import (
    init_network_params, pack_params, unpack_params,
    layer_sizes, get_batches, batched_predict
)

# -----------------------------------------------------------
# Entrenamiento con SWA (acumulación desde cierta época)
# -----------------------------------------------------------
def entrenar_con_swa(data, swa_start_epoch=30):
    xx, ff, nx, ny = data

    key = random.PRNGKey(0)
    params = init_network_params(layer_sizes, key)
    params = pack_params(params)
    step_size = 0.001
    num_epochs = 50
    batch_size = 32

    swa_sum = None
    swa_count = 0
    log_train = []

    for epoch in range(num_epochs):
        idxs = random.permutation(random.PRNGKey(epoch), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs=batch_size):
            grads = grad(lambda p, x, y: jnp.mean((batched_predict(p, x) - y)**2))(params, xi, yi)
            params = params - step_size * grads

        train_loss = jnp.mean((batched_predict(params, xx) - ff)**2)
        log_train.append(train_loss)

        if epoch >= swa_start_epoch:
            swa_sum = params if swa_sum is None else swa_sum + params
            swa_count += 1

        print(f"[SWA] Epoch {epoch+1} - Loss: {train_loss:.6f}")

    # Promedio de los parámetros
    swa_params = swa_sum / swa_count
    return params, swa_params, log_train, (nx, ny)

def datainit():
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
    return (xx, ff, nx, ny)

if __name__ == "__main__":
    data = datainit()
    params_final, params_swa, losses, (nx, ny) = entrenar_con_swa(data)

    xx, ff, _, _ = data

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Curva de pérdida - SWA")

    # Imagen objetivo
    plt.figure()
    plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')
    plt.title("Campo original")

    # Sin SWA
    pred_std = batched_predict(params_final, xx).reshape((nx, ny))
    plt.figure()
    plt.imshow(pred_std.T, origin='lower', cmap='jet')
    plt.title("Predicción final (sin SWA)")

    # Con SWA
    pred_swa = batched_predict(params_swa, xx).reshape((nx, ny))
    plt.figure()
    plt.imshow(pred_swa.T, origin='lower', cmap='jet')
    plt.title("Predicción final (SWA)")

    plt.show()