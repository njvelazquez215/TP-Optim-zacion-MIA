# -----------------------------------------------------------
# Punto 3.c del Trabajo Práctico - Visualización simplificada de superficie de pérdida
# -----------------------------------------------------------

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from nn_functions import init_network_params, pack_params, unpack_params, layer_sizes, batched_predict

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
    return xx, ff

def loss(p, x, y):
    preds = batched_predict(p, x)
    return jnp.mean((preds - y)**2)

if __name__ == "__main__":
    xx, ff = datainit()
    key1, key2, key3 = random.split(random.PRNGKey(0), 3)

    # 3 puntos en el espacio: inicial, entrenado, aleatorio
    p0 = pack_params(init_network_params(layer_sizes, key1))       # inicio
    p1 = p0 - 0.001 * jnp.ones_like(p0)  # simula un paso de entrenamiento
    p2 = pack_params(init_network_params(layer_sizes, key2))       # otro random

    alphas = jnp.linspace(-1, 2, 50)
    betas = jnp.linspace(-1, 2, 50)
    Z = jnp.zeros((len(alphas), len(betas)))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            p = p0 + a * (p1 - p0) + b * (p2 - p0)
            Z = Z.at[i, j].set(loss(p, xx, ff))

    plt.contourf(alphas, betas, Z.T, levels=50, cmap='viridis')
    plt.xlabel("Dirección p1 - p0")
    plt.ylabel("Dirección p2 - p0")
    plt.title("Visualización simplificada de la superficie de pérdida")
    plt.colorbar(label="Loss")
    plt.show()