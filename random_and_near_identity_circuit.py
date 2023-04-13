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
import mps_circuit_helpers as helpers

from tqdm import tqdm
from jax.config import config
config.update("jax_enable_x64", True)
from MPScumulant import MPS_c

N_ITS = 15000
LEARNING_RATE = 0.01
maxi_bond = 2
chi = 'BStest/BS_project-2-MPS'    
mps = MPS_c(12, max_bond_dim=maxi_bond)

def get_data_states(location, columns):
    """
    Given link, return bars and stripes data reshaped

    Args:
        location (string): location of the bars and stripes data
        columns (int): reshaping integer

    Returns:
        list: list of list of binary numbers
    """
    data = np.load(location)

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

def get_unitaries():
    """
    Get list of unitary matrices from MPS 

    Returns:
        list: list of unitaries
    """
    # chis = ['BStest/BS_project-2-MPS', 'BStest/BS_project-4-MPS', 'BStest/BS_project-8-MPS']
    # maxi_bonds = [2, 4, 8]
    
    mps.loadMPS(chi)
    for i in range(len(mps.matrices)):
        tn_core = mps.matrices[i]
        
    m_pad = helpers.pad_mps(mps)

    for i in range(len(m_pad.matrices)):
        tn_core = m_pad.matrices[i]
        
    m_pad.left_cano()
    unitary_list = []
    tn_cores = m_pad.matrices
    for site_idx in range(len(tn_cores)):
        tn_core = tn_cores[site_idx]
        u_mat = helpers.isometry_to_unitary(tn_core.reshape(-1, tn_core.shape[2]))
        u_mat = pnp.array(u_mat, requires_grad=True)
        u_mat = jnp.array(u_mat)
        unitary_list.append(u_mat)
        
    # unitary_list = pnp.array(unitary_list)
    return unitary_list

def get_mps_circ_extended(weights, n_wires=12, shots=None):
    """
    Create quantum circuit with unitaries from MPS extended with Special Unitaries

    Args:
        weights (list): list of weights
        mps (Matrix product State): matrix product state
        n_wires (int, optional): number of wires. Defaults to 12.
        shots (_type_, optional): number of shots. Defaults to None.

    Returns:
        qml.QNode : Initialized quantum circuit
    """
    unitary_list = get_unitaries()
        
    truncated_unitary_list = unitary_list[1:]
    n_wires = len(truncated_unitary_list) + 1

    dev = qml.device("default.qubit", wires=n_wires, shots=shots)
    @qml.qnode(dev, interface="jax")
    def qnode():
        wires_connected = []
        for wire in range(n_wires-1, -1, -1):
            unitary = unitary_list[wire]
            n_qubits = int(np.log2(unitary.shape[0]))
            u_wires = [wire] + list(range(wire-1, wire-n_qubits, -1))
            u_wires.reverse()
            qml.QubitUnitary(unitary, wires=u_wires)
            wires_connected.append(u_wires)

        wires_connected_len_2 = set()
        for w in wires_connected:
            for i in range(len(w) - 1):
                for j in range(i, len(w)):
                    wires_connected_len_2.add((w[i], w[j]))
                    
        for x in range(11):
            for j in range(x + 1, 12):
                if (x, j) not in wires_connected_len_2 and (j, x) not in wires_connected_len_2:
                    qml.SpecialUnitary(weights[i], wires=[x, j])
        if shots is not None:
            return qml.sample()
        
        return qml.probs(wires=list(range(n_wires)))
    
    return qnode

@jax.jit
def loss_random_near_unitary( weights):
    """
    Loss function for randomly intialized and near unitary initialized weights

    Args:
        weights (list): list of weights

    Returns:
        _type_: KL divergence
    """
    probs = get_random_initialized_circuit(weights)()
    filter_qc_probs = metrics.filter_probs(probs, get_data_states("BStest/b_s_4_3.npy", 12))
    return metrics.kl_divergence_synergy_paper(22, filter_qc_probs)

@jax.jit
def loss_mps_extended(weights):
    """
    Loss function for randomly intialized and near unitary initialized weights

    Args:
        weights (list): list of weights

    Returns:
        _type_: KL divergence
    """
    
    probs = get_mps_circ_extended(weights)()
    filter_qc_probs = metrics.filter_probs(probs, get_data_states("BStest/b_s_4_3.npy", 12))
    return metrics.kl_divergence_synergy_paper(22, filter_qc_probs)


def plot_KL_divergence( loss_track, title):
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

    
def train_model(weights, mps=None, circuit_type="random_near_unitary"):
    """
    Train on circuit using weights

    Args:
        weights (list): list of weights

    Returns:
        tuple: a tuple of the updated weights and kl divergence for each iteration
    """
    loss_track = []
    N_ITS = 15000
    LEARNING_RATE = 1e-6   
    opt_exc = optax.adam(LEARNING_RATE)
    opt_state = opt_exc.init(weights)

    for it in tqdm(range(N_ITS)):
        grads = None
        if circuit_type == "random_near_unitary":
            grads = jax.grad(loss_random_near_unitary)(weights)
            updates, opt_state = opt_exc.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)
            loss_track.append(loss_random_near_unitary(weights))
        else:
            grads = jax.grad(loss_mps_extended)(weights)
            updates, opt_state = opt_exc.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)
            loss_track.append(loss_mps_extended(weights))
    return weights, loss_track

def count_number_of_weights(unitary_list, n_wires=12):
    """
    Count number of weights needed

    Args:
        unitary_list (list): list of unitary matrices
        n_wires (int, optional): number of wires. Defaults to 12.

    Returns:
        int: number of weights needed
    """
    total_num_weights = 0
    wires_connected = []
    for wire in range(n_wires-1, -1, -1):
        n_qubits = int(np.log2(unitary_list[wire].shape[0]))
        u_wires = [wire] + list(range(wire-1, wire-n_qubits, -1))
        u_wires.reverse()
        wires_connected.append(u_wires)
        
    wires_connected_len_2 = set()
    for w in wires_connected:
        for i in range(len(w) - 1):
            for j in range(i, len(w)):
                wires_connected_len_2.add((w[i], w[j]))
                
    for x in range(11):
        for j in range(x + 1, 12):
            if (x, j) not in wires_connected_len_2 and (j, x) not in wires_connected_len_2:
                total_num_weights += 1
                
    return total_num_weights
            
