from itertools import permutations

import numpy as np

k = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])


def softmax_prime(softmax_output: np.ndarray):
    return np.einsum(
        "ji,jk->ijk", softmax_output, np.eye(softmax_output.shape[0])
    ) - np.einsum("ji,ki->ijk", softmax_output, softmax_output)


softmax_grad = softmax_prime(k)
delta = k
# print(softmax_grad)
# print(delta)
softmax_input_grad = np.einsum("kij,jk->ik", softmax_grad, delta)
print(softmax_input_grad)


# for s in permutations(["i", "j", "k"]):
#     print(s)
#     print(np.einsum(f"{''.join(s)},jk->ik", softmax_grad, delta))
