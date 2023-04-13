"""
CPEN 400Q work

Pennylane implementation of quantum circuits for randomly initialized and near identity initialized Special Unitaries

Author : @mushahidkhan123
"""

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import optax
import jax
import mps_circuit_helpers
import mps_circuit
import metrics

from tqdm import tqdm
from jax.config import config
config.update("jax_enable_x64", True)
from MPScumulant import MPS_c

N_ITS = 15000
LEARNING_RATE = 0.01

def get_data_states(location, columns):
    """
    Given link, return bars and stripes data reshaped

    Args:
        location (string): location of the bars and stripes data
        columns (int): reshaping integer

    Returns:
        list: list of list of binary numbers
    """
    data = np.load('BStest/b_s_4_3.npy')

    return data.reshape(-1, columns).astype(np.int8)

def get_random_initialized_circuit(weights, shots = None, wires = 12):
    """
    Given a list of randomly initialized initialize a quantum
    circuit

    Args:
        weights (list): List of weights

    Returns:
        qml.QNode : Initialized quantum circuit
    """
    dev = qml.device("default.qubit", wires=wires, shots=shots)
    
    @qml.qnode(dev, interface="jax")
    def qnode():
        for i in range(11):
            qml.SpecialUnitary(weights[i], wires=[i, i + 1])
            qml.SpecialUnitary(weights[i + 11], wires=[i, i  + 1])
        
        for x in range(wires - 1):
            for j in range(x + 1, wires):
                qml.SpecialUnitary(weights[i], wires=[x, j])
                i += 1

        if shots is not None:
            return qml.sample()
        
        return qml.probs(wires=list(range(wires)))
    
    return qnode

@jax.jit
def loss(weights):
    probs = get_random_initialized_circuit(weights)()
    filter_qc_probs = metrics.filter_probs(probs, get_data_states("Stest/b_s_4_3.npy", 12))
    return metrics.kl_divergence_synergy_paper(22, filter_qc_probs)

def plot_KL_divergence(loss_track, title):
    """
    Plot the KL divergence of model

    Args:
        loss_track (list): list of KL divergence for each iteration
    """
    plt.plot(loss_track)
    plt.title(f"Training KL - Divergence:{title}")
    plt.xlabel("Iterations")
    plt.ylabel("KL - Divergence")
    plt.yscale("log")
    
    
def train_model(weights):
    """
    Train on circuit using weights

    Args:
        weights (list): list of weights

    Returns:
        tuple: a tuple of the updated weights and kl divergence for each iteration
    """
    loss_track = []
    opt_exc = optax.adam(LEARNING_RATE)
    opt_state = opt_exc.init(weights)

    for it in tqdm(range(N_ITS)):
        grads = jax.grad(loss)(weights)
        updates, opt_state = opt_exc.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        loss_track.append(loss(weights))
    return weights, loss_track