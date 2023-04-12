"""
CPEN 400Q work

Pennylane implementation of quantum circuits constructed using the classically
trained MPS

Author : @abhishekabhishek
"""
import pennylane as qml
import numpy as np
from mps_circuit_helpers import is_unitary


def mps_unitaries_to_circuit(mps_unitaries, shots: int = None):
    """
    Given a list of unitaries extracted from an MPS, initialize a quantum
    circuit with the trained parameters

    Args:
        mps_unitaries (list): List of multi-qubit unitaries constructed from
            the MPS using the helper functions in mps_circuit_helpers.py

    Returns:
        qml.QNode : Initialized quantum circuit corresponding to the trained
            MPS
    """
    # check all matrices in the list are actually unitaries
    for idx, unitary in enumerate(mps_unitaries):
        assert is_unitary(unitary), \
            f"the matrix at idx {idx} in the list is not unitary"
        assert unitary.shape[0]%2 == 0, \
            f"the unitary size at idx {idx} is not a power-of-two : \
                {unitary.shape}"

    n_wires = len(mps_unitaries)
    dev = qml.device("default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        # starting from wire 0, apply the multi-qubit unitaries in the list
        # in a staircase format
        for wire in range(n_wires-1, -1, -1):
            unitary = mps_unitaries[wire]
            n_qubits = int(np.log2(unitary.shape[0]))
            u_wires = [wire] + list(range(wire-1, wire-n_qubits, -1))
            u_wires.reverse()
            qml.QubitUnitary(unitary, wires=u_wires)

        # return bitstring samples if number of shots specified
        if shots is not None:
            return qml.sample()
        # else return the probs of bitstrings
        return qml.probs(wires=range(n_wires))

    return circuit
