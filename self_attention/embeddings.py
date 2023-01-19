"""
Modul for loading word embeddings into a dictionary in the form: token->NDarray

Eager loads the dataset, which can take a few seconds depending on the embedding dimension.
"""

import numpy as np
from numpy.typing import NDArray
from functools import partial ; l2_norm = partial(np.linalg.norm, ord=2)


def _load_glove_model(file_path) -> dict[str, NDArray]:
    # Adapted from:
    # https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

    print("Loading Glove Model: ", file_path)
    glove_model = {}

    with open(file_path, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding

    print(f"{len(glove_model)} words loaded!")
    return glove_model


def embed(token: str) -> NDArray:
    """Embed a token into the embedding (word-vector) space."""
    return _GLOVE[token]


def unembed(vector: NDArray) -> str:
    """Find the closest vector.
    
    XXX: slow.
    """
    smallest_distance = float("inf")
    
    for k, v in _GLOVE.items():
        distance = np.dot(vector, v) / (l2_norm(vector) * l2_norm(v))

        if distance < smallest_distance:
            smallest_distance = distance
            most_similar_word = k
    
    return most_similar_word


_GLOVE = _load_glove_model("data/embeddings/glove.6B.50d.txt")
