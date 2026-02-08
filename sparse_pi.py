import numpy as np
import torch


def sparsify_pi(pi):
    """Convert a dense pi vector to sparse (indices, values) format.

    Accepts a list, numpy array, or already-sparse tuple.
    Returns (indices as uint16, values as float16).
    """
    if isinstance(pi, tuple):
        return pi  # already sparse
    arr = np.asarray(pi)
    nonzero = np.nonzero(arr)[0]
    return (nonzero.astype(np.uint16), arr[nonzero].astype(np.float16))


def dense_pis_from_sparse(sparse_pis, action_size):
    """Reconstruct a batch of dense pi tensors from sparse format.

    Args:
        sparse_pis: list of (indices, values) tuples
        action_size: total action space size

    Returns:
        torch.FloatTensor of shape (batch_size, action_size)
    """
    batch_size = len(sparse_pis)
    dense = torch.zeros(batch_size, action_size)
    for i, pi in enumerate(sparse_pis):
        if isinstance(pi, tuple):
            idxs, vals = pi
            dense[i, idxs] = torch.as_tensor(vals, dtype=torch.float32)
        else:
            dense[i] = torch.as_tensor(np.asarray(pi), dtype=torch.float32)
    return dense
