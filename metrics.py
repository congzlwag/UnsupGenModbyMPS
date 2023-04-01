"""
CPEN 400Q work

Helper functions to calculate the metrics such as KL-divergence of the models

Author : @abhishekabhishek
"""
import numpy as np
from pennylane import numpy as pnp


def filter_probs(probs, data):
    """
    filter the 2**n_qubits bitstring probability values to use only the ones
    from the bitstrings that show up in our dataset

    Args:
        probs (qml.np.tensor) : Bitstring probabilities output by the qnode
        data (np.ndarray) : (n_samples, n_features) matrix of our training
            dataset

    Returns:
        probs (qml.np.tensor) : Filtered probability vector containing values
            only for bitstrings that show up in the dataset
    """
    assert len(data.shape) == 2, f"the data matrix is not 2-dimensional : \
        {data.shape}"
    idxs = np.dot(data, 2**np.arange(data.shape[1])[::-1])
    return probs[idxs]


def kl_divergence_estimate(probs):
    """
    analytically calculate the kl-divergence using the probabilities assigned
    by the QCBM to the dataset bitstring using Born's rule

    Args:
        probs (qml.np.tensor) : Model probabilities for each of the bitstrings
            in our dataset

    Returns:
        (qml.np.tensor) : Floating point value of an estimate KL-divergence

    # TODO can this be used as a cost function to train the QCBM
    """
    # Uniform empirical probability of the bitstrings in the dataset
    p_data = pnp.tensor(1./len(probs), requires_grad=False)
    log_p_over_q = pnp.log(probs) - pnp.log(p_data)
    return pnp.dot(probs, log_p_over_q)


def kl_divergence(probs, data):
    """
    compute the kl divergence over the entire space of the bitstrings

    Args:
        probs (qml.np.tensor) : Output of the qnode - probabilities for each
            basis state

    Returns:
        (qml.np.tensor) : Floating point value of the KL-divergence
    """
    # p_data with epsilon added to avoid log(0) error
    p_data = pnp.zeros(len(probs), requires_grad=False) + 1e-16
    idxs = np.dot(data, 2**np.arange(data.shape[1])[::-1])
    p_data[idxs] = 1./data.shape[0]
    log_p_over_q = pnp.log(probs) - pnp.log(p_data)
    return pnp.dot(probs, log_p_over_q)
