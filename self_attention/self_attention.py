"""Module for applying the self-attention mechanism to a sequence of vectors,
for instance word-vectors.
"""
import numpy as np
from numpy.typing import NDArray


def _softmax(weights: np.array) -> NDArray:
    """Softmax the elements of the input weights such that they sum to 1."""

    # XXX: not-optimized at all
    scaled_weights = [np.exp(weight) / sum(np.exp(weights)) for weight in weights]
    return np.array(scaled_weights)
        


def apply(x: NDArray) -> NDArray:
    """Apply the self-attention mechanism to the input sequence to
    generate an output sequence.
    """
    w = np.dot(x.T, x)
    w /= np.sqrt(x.shape[0])

    # w_prime should be symmetric
    # two tokens should pay equal attention to each other.
    #
    # e.g: equal w_prime_(ij) should equal w_prime_(ji)
    assert (w == w.T).all()

    # w' -> w 
    for i, row in enumerate(w):
        # Softmax each row.
        w[i] = _softmax(row)

    return np.dot(x, w)


if __name__ == "__main__":
    x1 = np.array([1., 0., 0.])
    x2 = np.array([1., 0., 0.])
    x3 = np.array([0., 1., 1.])
    x = np.stack((x1, x2, x3), axis=1)
    
    y = apply(x)
    # The attention of x1 and x2 should be equal, they are the same vectors!
    assert y[0].all() == y[1].all()