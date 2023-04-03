"""
CPEN 400Q work

Helper functions to map a trained MPS model to a quantum circuit using the
simplest multi-qubit gate mapping

Author : @abhishekabhishek
"""
import numpy as np
import scipy

from MPScumulant import MPS_c


def is_left_isometry(mat):
    """
    check if a matrix is a left isometry i.e. mat^{+}*mat = I

    Args:
        mat (np.ndarray): a rectangular matrix (a square matrix is special 
            case)

    Returns:
        bool: boolean
    """
    assert len(mat.shape) == 2, 'the matrix needs to be 2-dimensional'
    return np.allclose(
        np.eye(mat.shape[1]),
        np.matmul(mat.conj().T, mat)
    )


def is_right_isometry(mat):
    """
    check if a matrix is a right isometry i.e. mat*mat^{+} = I

    Args:
        mat (np.ndarray): a rectangular matrix (a square matrix is special 
            case)

    Returns:
        bool: boolean
    """
    assert len(mat.shape) == 2, 'the matrix needs to be 2-dimensional'
    return np.allclose(
        np.eye(mat.shape[0]),
        np.matmul(mat, mat.conj().T)
    )


def is_unitary(mat):
    """
    check if a matrix is a unitary i.e. mat^{+}*mat = mat*mat^{+} = I

    Args:
        mat (np.ndarray): a rectangular matrix (a square matrix is special 
            case)

    Returns:
        bool: boolean
    """
    assert len(mat.shape) == 2, 'the matrix needs to be 2-dimensional'
    if mat.shape[0] != mat.shape[1]:
        return False
    else:
        return (is_left_isometry(mat) and is_right_isometry(mat))


def are_isometries(mps, reshape_axis: int = 0):
    """
    check if the core tensors in the MPS are right or left isometries after
    merging the open axis with either the left (0) or right (2) virtual axis

    Args:
        mps (MPScumulant.MPS_c): MPS object containing the core tensors and
            bond dimensions.
        reshape_axis (int): The virtual axis of the core tensors along which
            to merge the physical axis (i.e. the open edge), should be either 0
            or 2 i.e. merge along the left or right virtual axis respectively.

    Output:
        For each core tensor in the MPS, prints its
        (idx, shape, is it a left isometry, is it a right isometry)

    Returns:
        None
    """
    assert reshape_axis in [0, 2], 'reshape axis should either be 0 (merge \
        along the left virtual axis or 2 (merge along the right virtual axis)'

    print('idx, core tensor shape, left isometry, right isometry')

    for i, tn_core in enumerate(mps.matrices):
        # convert the order-3 core tensor to a matrix
        # merge along the left virtual axis
        if reshape_axis == 0:
            core_mat = tn_core.reshape(-1, tn_core.shape[2])
        else:
            core_mat = tn_core.reshape(tn_core.shape[0], -1)

        print(i, tn_core.shape, is_left_isometry(core_mat),
              is_right_isometry(core_mat))


def pad_mps(mps):
    """
    pads (in-place) all the core tensors of an MPS to have bond dimensions
    which are powers of two (this is required when we want to map these core
    tensors to multi-qubit gates)

    Args:
        mps (MPScumulant.MPS_c): MPS object containing the core tensors and
            bond dimensions.

    Output:
        For each core tensor in the MPS, prints its idx, its new size after the
        padding, the updated bond dimension to the left of it

    Returns:
        mps_pad (MPScumulant.MPS_c): MPS object with padded core tensors
    """
    # extract the core tensors and bond dimensions of the original MPS
    tn_cores, bond_dims = mps.matrices.copy(), \
        mps.bond_dimension.copy()

    # create a new MPS - this is needed for the left-canonicalization method
    # to work correctly
    mps_pad = MPS_c(len(bond_dims))

    # iterate all over core tensors and make the bond dimensions b/w them into
    # powers of two
    print("idx, shape of the padded tensor, updated bond dimension")
    for i in range(1, len(tn_cores)-1):
        tn_core = tn_cores[i]
        pad_width_list = []
        for _, dim in enumerate(tn_core.shape):
            if dim % 2 == 0:
                pad_width_list.append((0, 0))
            else:
                pad_width_list.append((0, 1))
        tn_core = np.pad(tn_core, pad_width_list, mode='constant',
                         constant_values=0)

        tn_cores[i] = tn_core
        bond_dims[i-1] = tn_core.shape[0]
        print(f"i = {i}, {tn_core.shape}, {tn_core.shape[0]}")

    # update the tensors and bond dims of the MPS with the padded tensors and
    # the updated corresponding bond dimensions
    mps_pad.matrices, mps_pad.bond_dimension = tn_cores, bond_dims

    return mps_pad


def isometry_to_unitary(mat):
    """
    convert an isometry to a unitary

    Args:
        mat (np.ndarray) : a rectangular matrix

    Returns:
        u_mat (np.ndarray) : a square unitary matrix constructed from the 
            isometry
    """
    # special case where the input matrix is already a unitary
    if is_unitary(mat):
        return mat

    assert is_left_isometry(mat) or is_right_isometry(mat), \
        'the input matrix is not an isometry'

    assert len(mat.shape) == 2, 'the matrix is not 2-dimensional'

    # TODO: not sure if this method works as-is for both left and right
    # isometries

    # get basis vectors from the null (kernel) space of mat^{+}

    n_rows, n_cols = mat.shape
    x_mat = scipy.linalg.null_space(mat.conj().T)

    if n_rows < n_cols:
        u_mat = np.vstack((mat, x_mat))
    else:
        u_mat = np.hstack((mat, x_mat))

    return u_mat


def get_mps_unitaries(mps):
    """
    get a list of multi-qubit unitaries from an MPS corresponding to unitaries
    which can be used to build the staircase circuit in qml

    Steps that need to performed:
        1. pad the MPS core tensors with zeros to ensure the bond dimensions
           are multiples of 2
        2. left-canonicalize the MPS to ensure all core tensors are left-
           isometries
        3. starting from the tensor at site 1 (0-indexed), convert the core
           tensor to a unitary and add to the list
        4. repeat step 3. to the right towards the last site of the MPS

    Args:
        mps (MPScumulant.MPS_c): MPS object containing the core tensors and
            bond dimensions.

    Returns;
        unitary_list (List): List of multi-qubit unitaries starting from the
        multi-qubit unitaries to be applied to wire 0 and [1, ..., n] wires,
        wire 1 and [2, ..., m] wires and so on.  
    """
    # step 1 : pad the MPS using the in-place padding helper function
    print('padding the mps core tensors')
    mps_pad = pad_mps(mps)

    # step 2 : left canonicalize the MPS
    print('left canonicalizing the padded MPS')
    mps_pad.left_cano()

    # step 3-4 : convert core tensors (which should now be left isometries) to
    # unitaries and add to the list
    unitary_list = []
    tn_cores = mps_pad.matrices

    #TODO set this back to 1 for normal behaviour
    start_idx = 1

    for site_idx in range(start_idx, len(tn_cores)):
        tn_core = tn_cores[site_idx]

        # this step assumes that the core tensor is a left isometry
        u_mat = isometry_to_unitary(tn_core.reshape(-1, tn_core.shape[2]))
        unitary_list.append(u_mat)

    return unitary_list
