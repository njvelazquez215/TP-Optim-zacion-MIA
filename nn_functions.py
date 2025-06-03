import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

import matplotlib.pyplot as plt

def pack_params(params):
    """Pack parameters into a single vector."""
    return jnp.concatenate([jnp.ravel(w) for w, _ in params] +
                             [jnp.ravel(b) for _, b in params])

layer_sizes = [2, 64, 64, 1]

def unpack_params(params):
    """Unpack parameters from a single vector."""
    weights = []
    for i in range(len(layer_sizes) - 1):
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        to_unpack, params = params[:weight_size], params[weight_size:]
        weights.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1], layer_sizes[i]))

    biases = []
    for i in range(len(layer_sizes) - 1):
        bias_size = layer_sizes[i + 1]
        to_unpack, params = params[:bias_size], params[bias_size:]
        biases.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1]))

    params = [(w, b) for w, b in zip(weights, biases)]
    return params

def random_layer_params(m, n, key, scale=1e-2):
    ''' Randomly initialize weights and biases for a dense neural network layer '''
    w_key, b_key = random.split(key)
    scale = jnp.sqrt(6.0 / (m + n))
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    # return jnp.ones((n, m)), jnp.zeros((n,))

def init_network_params(sizes, key):
    ''' Initialize all layers for a fully-connected neural network with sizes "sizes" '''
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def predict(params, coord):
    params = unpack_params(params)
    hidden = coord
    for w, b in params[:-1]:
        outputs = jnp.dot(w, hidden) + b
        hidden = nn.tanh(outputs)

    final_w, final_b = params[-1]
    output = jnp.dot(final_w, hidden) + final_b
    return output

batched_predict = vmap(predict, in_axes=(None, 0))

@jit
def scalar_predict(params, coord):
    return predict(params, coord)[0]

@jit
def grad_predict(params, coord):
    return grad(scalar_predict, argnums=1)(params, coord)

batched_grad_predict = vmap(grad_predict, in_axes=(None, 0))

@jit
def loss(params, coord, target, gradi, lmbd, lmbd_grad):
    preds = batched_predict(params, coord)
    loss_data = jnp.mean(jnp.square(preds - target))
    if not lmbd is None:
        weights = unpack_params(params)
        l2_penalty = sum(jnp.sum(w**2) for w, _ in weights)
        loss_data += lmbd * l2_penalty
    if not lmbd_grad is None:
        grad_preds = batched_grad_predict(params, coord)
        loss_data += lmbd_grad * jnp.mean(jnp.square(grad_preds - gradi))
    return loss_data

# @jit
# def loss_l2(params, coord, target, lmbd):
#     preds = batched_predict(params, coord)
#     loss_data = jnp.mean(jnp.square(preds - target))
#     weights = unpack_params(params)
#     l2_penalty = sum(jnp.sum(w**2) for w, _ in weights)
#     return loss_data + lmbd * l2_penalty
@jit
def update_sgd(params, x, y, gradi, step_size, aux, t, lmbd, lmbd_grad):
    grads  = grad(loss)(params, x, y, gradi, lmbd, lmbd_grad)
    params = params - step_size * grads
    return params, aux

@jit
def update_rmsprop(params, x, y, gradi, step_size, aux, t, lmbd, lmbd_grad):
    beta = 0.9
    grads = grad(loss)(params, x, y, gradi, lmbd, lmbd_grad)
    aux = beta * aux + (1 - beta) * jnp.square(grads)
    step_size = step_size / (jnp.sqrt(aux) + 1e-8)
    params = params - step_size * grads 
    return params, aux

@jit
def update_adam(params, x, y, gradi, step_size, aux, t, lmbd, lmbd_grad):
    sk, rk, beta1, beta2 = aux
    grads  = grad(loss)(params, x, y, gradi, lmbd, lmbd_grad)
    sk = beta1 * sk + (1 - beta1) * grads
    rk = beta2 * rk + (1 - beta2) * jnp.square(grads)
    svk = sk / (1 - beta1 ** t)
    rvk = rk / (1 - beta2 ** t)
    step_size = step_size / (jnp.sqrt(rvk) + 1e-8)
    params = params - step_size * svk
    aux = (sk, rk, beta1, beta2)
    return params, aux

# def update_pswa(params, x, y, gradi, step_size, aux, t, lmbd, lmbd_grad):
#     update, optimAux, schAux = aux

#     a1, a2, cCLR, cSWA, nSWA, paramsSWA = schAux

#     # CLR
#     k = 1 / cCLR * (((t - 1) % cCLR) + 1)
#     step_size = (1 - k) * a1 + k * a2
#     if k == 1:  # Cuando el CLR llega al menor paso se realiza un promedio.
#         print('Se promedia')
#         paramsSWA = (paramsSWA * nSWA+ params)/(nSWA + 1)
#         nSWA += 1
#         if nSWA == cSWA:    # Cuando ya se promediaron cSWA pesos (un ciclo), se fuerzan los pesos promediados en el entrenamiento,
#                             # se reinician el conteo de pesos promedios (nSWA) y los pesos promediados (paramsSWA).
#             print('SWA')
#             params = paramsSWA
#             nSWA = 0
#             paramsSWA = 0

#     schAux = a1, a2, cCLR, cSWA, nSWA, paramsSWA

#     # Optimizador
#     params, optimAux = update(params, x, y, step_size, optimAux, t)

#     aux = (update, optimAux, schAux)

#     return params, aux


def get_batches(x, y, grad_ff,  bs):
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs], grad_ff[i:i+bs, :]

@jit
def sch_exponential(step_size, schAux, t):
    r, = schAux
    return step_size * 10 ** (-1 / r)

@jit
def sch_power(step_size, schAux, t):
    step_size0, r, c = schAux
    return step_size0 * (1 + t / r) ** - c

@jit
def sch_CLR(step_size, schAux, t):
    a1, a2, c = schAux
    k = 1 / c * (((t - 1) % c) + 1)
    return (1 - k) * a1 + k * a2
    
def activaciones(params, coord):
    params = unpack_params(params)
    hidden = coord
    activaciones = []
    for w, b in params[:-1]:
        outputs = jnp.dot(w, hidden) + b
        hidden = nn.tanh(outputs)
        activaciones.append(hidden)
    # final_w, final_b = params[-1]
    # output = jnp.dot(final_w, hidden) + final_b
    return activaciones

batched_activaciones = vmap(activaciones, in_axes=(None, 0))