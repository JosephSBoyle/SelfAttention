import numpy as np
import embeddings, self_attention


# Example of a sentence with each token stemmed.
input_ = (
    "the",
    "rat",
    "likes",
    "eat",   # stemmed from 'eating' to 'eat'.
    "pizza",
)

embedded_input = np.transpose(np.array([embeddings.embed(x) for x in input_]))
# transpose to make each column represent a word (standard)

embedded_output = self_attention.apply(embedded_input)
embedded_output = self_attention.apply(embedded_output)

# transpose to iterate through the colums of y (the rows)
unembedded_output = [embeddings.unembed(y) for y in embedded_output.T]
print(input_)
print(unembedded_output)