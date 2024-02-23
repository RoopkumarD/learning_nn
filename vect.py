import numpy as np

training_data_x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
training_data_y = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def parameter_initialize():
    W1 = np.random.rand(2, 2) - 0.5
    B1 = np.zeros((2, 1))
    W2 = np.random.rand(1, 2) - 0.5
    B2 = np.zeros((1, 1))
    return W1, B1, W2, B2


W1, B1, W2, B2 = parameter_initialize()
epoch = 10000
learn_rate = 0.1


def feed_forward(x):
    z1 = np.dot(W1, x) + B1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + B2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def total_err(y, res):
    return np.sum((y - res) ** 2) * 0.5


for ep in range(epoch):
    x = training_data_x.transpose()

    z1, a1, z2, a2 = feed_forward(x)

    delta = (a2 - training_data_y) * sigmoid_prime(z2)
    delta_b2 = np.sum(delta, axis=1).reshape((-1, 1))
    delta_w2 = np.dot(delta, a1.transpose())
    delta2 = np.dot(W2.transpose(), delta) * sigmoid_prime(z1)
    delta_b1 = np.sum(delta2, axis=1).reshape((-1, 1))
    delta_w1 = np.dot(delta2, x.transpose())

    print(f"Epoch {ep} -> Err {total_err(training_data_y, a2)/4}")

    # updating the error
    B2 -= delta_b2 * learn_rate
    W2 -= delta_w2 * learn_rate
    B1 -= delta_b1 * learn_rate
    W1 -= delta_w1 * learn_rate


testing_data = [
    [np.array([1, 1]), 0],
    [np.array([1, 0]), 1],
    [np.array([0, 1]), 1],
    [np.array([0, 0]), 0],
]

# preds
for x, y in testing_data:
    x = np.resize(x, new_shape=(2, 1))
    _, _, _, res = feed_forward(x)
    print(f"{x.flatten()} result is {res.round().flatten()}")
