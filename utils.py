import numpy as np


def relu(input: np.ndarray):
    return np.maximum(input, 0)


def relu_prime(input: np.ndarray):
    return (input > 0) * 1


def sigmoid(input: np.ndarray):
    d = np.exp(-input)
    return 1 / (1 + d)


def sigmoid_prime(input: np.ndarray):
    s = sigmoid(input)
    return s * (1 - s)


# def softmax(linear_value: np.ndarray):
#     e = np.exp(linear_value)
#     res = e / np.sum(e, axis=0)
#     return res


def softmax(x):
    # Subtract the maximum value in each row (axis=0) for numerical stability
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))

    # Normalize by the sum of exponentials along each row
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


# for one array of softmax output
# def softmax_prime(softmax_output: np.ndarray):
#     return np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)


# vectorised
def softmax_prime(softmax_output: np.ndarray):
    return np.einsum(
        "ji,jk->ijk", softmax_output, np.eye(softmax_output.shape[0])
    ) - np.einsum("ji,ki->ijk", softmax_output, softmax_output)
