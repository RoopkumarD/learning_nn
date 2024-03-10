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


# convolution
def image_convolution(kernel: np.ndarray, image: np.ndarray, stride: int):
    row, col, kernel_size = len(image), len(image[0]), len(kernel)
    result_row = row - kernel_size + 1
    result_col = col - kernel_size + 1
    result = np.zeros((result_row, result_col))

    for r in range(0, result_row, stride):
        for c in range(0, result_col, stride):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    result[r][c] += image[r + i][c + j] * kernel[i][j]

    return result


def max_pooling(pool_size: int, feature_map: np.ndarray):
    final_image_size = int(len(feature_map) / pool_size)
    row, col = pool_size, pool_size
    result = np.zeros((final_image_size, final_image_size))
    pooled_elem = []

    for r in range(final_image_size):
        for c in range(final_image_size):
            temp_row = r * pool_size
            temp_col = c * pool_size
            max_index = np.argmax(
                feature_map[temp_row : temp_row + row, temp_col : temp_col + col]
            )
            temp = int(max_index / col)
            result[r][c] = feature_map[
                temp_row : temp_row + row, temp_col : temp_col + col
            ][temp][max_index - temp * col]
            pooled_elem.append((temp + temp_row, temp_col + max_index - temp * col))

    result = result.reshape((-1, 1))

    return result, pooled_elem
