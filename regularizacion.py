# -----------------------------------------------------------
# Punto 3.a del Trabajo Práctico - Regularización L2
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
# Función de pérdida con regularización L2 (lambda > 0)
# -----------------------------------------------------------
@jit
def loss_l2(params, coord, target, lmbd):
    preds = batched_predict(params, coord)
    loss_data = jnp.mean(jnp.square(preds - target))  # MSE
    weights = unpack_params(params)
    l2_penalty = sum(jnp.sum(w**2) for w, _ in weights)
    return loss_data + lmbd * l2_penalty

# -----------------------------------------------------------
# Entrenamiento con o sin regularización
# -----------------------------------------------------------
def entrenar(data, use_regularization=False, lmbd=0.001):
    xx, ff, nx, ny = data

    key = random.PRNGKey(0)
    params = init_network_params(layer_sizes, key)
    params = pack_params(params)
    step_size = 0.001
    num_epochs = 50
    batch_size = 32

    log_train = []

    for epoch in range(num_epochs):
        idxs = random.permutation(random.PRNGKey(epoch), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs=batch_size):
            if use_regularization:
                grads = grad(loss_l2)(params, xi, yi, lmbd)
            else:
                grads = grad(lambda p, x, y: jnp.mean((batched_predict(p, x) - y)**2))(params, xi, yi)
            params = params - step_size * grads

        train_loss = loss_l2(params, xx, ff, lmbd) if use_regularization else \
                     jnp.mean((batched_predict(params, xx) - ff)**2)
        log_train.append(train_loss)
        print(f"[{'REG' if use_regularization else 'NOREG'}] Epoch {epoch+1} - Loss: {train_loss:.6f}")

    return params, log_train, (nx, ny)

# -----------------------------------------------------------
# MAIN: Entrena y compara con vs sin regularización
# -----------------------------------------------------------
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

    # Entrenamiento sin regularización
    params_no_reg, loss_no_reg, (nx, ny) = entrenar(data, use_regularization=False)

    # Entrenamiento con regularización L2
    params_reg, loss_reg, _ = entrenar(data, use_regularization=True, lmbd=1e-4)

    # Comparar curvas de pérdida
    plt.figure()
    plt.plot(loss_no_reg, label="Sin regularización")
    plt.plot(loss_reg, label="Con L2 (λ=1e-4)")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Comparación: Regularización L2")
    plt.legend()
    plt.grid(True)

    # Campo objetivo
    xx, ff, _, _ = data
    plt.figure()
    plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')
    plt.title("Campo original")

    # Predicción sin regularización
    pred_no_reg = batched_predict(params_no_reg, xx).reshape((nx, ny))
    plt.figure()
    plt.imshow(pred_no_reg.T, origin='lower', cmap='jet')
    plt.title("Predicción sin regularización")

    # Predicción con regularización
    pred_reg = batched_predict(params_reg, xx).reshape((nx, ny))
    plt.figure()
    plt.imshow(pred_reg.T, origin='lower', cmap='jet')
    plt.title("Predicción con regularización L2")

    plt.show()