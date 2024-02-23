import numpy as np

from utils import sigmoid, sigmoid_prime


class Network:
    def __init__(self, nodes_num: np.ndarray):
        self.num_layers = len(nodes_num)

        self.weights: list[np.ndarray] = [
            np.random.rand(nodes_num[i + 1], nodes_num[i]) - 0.5
            for i in range(len(nodes_num) - 1)
        ]
        self.bias: list[np.ndarray] = [
            np.zeros((nodes_num[i], 1)) for i in range(1, len(nodes_num))
        ]

    def feed_forward(self, input: np.ndarray):
        z = np.array(input)
        for w, b in zip(self.weights, self.bias):
            z = sigmoid(np.dot(w, z) + b)
        return z

    def cost_function(self, input: np.ndarray, y):
        res = self.feed_forward(input)
        return np.sum((y - res) ** 2) * 0.5

    def train(
        self,
        training_x: np.ndarray,
        training_y: np.ndarray,
        epoch: int,
        learn_rate: float,
        momentum_const: float,
    ):
        # taking every input together
        velocity_dw = [np.zeros(w.shape) for w in self.weights]
        velocity_db = [np.zeros(b.shape) for b in self.bias]
        batch_num = len(training_x)
        for ep in range(epoch):
            z = training_x.transpose()
            zs = []
            activations = [np.array(z)]

            for w, b in zip(self.weights, self.bias):
                z = np.dot(w, z) + b
                zs.append(z)
                z = sigmoid(z)
                activations.append(z)

            delta = (activations[-1] - training_y) * sigmoid_prime(zs[-1])
            velocity_db[-1] = momentum_const * velocity_db[-1] - np.sum(
                delta, axis=1
            ).reshape((-1, 1)) * (learn_rate / batch_num)
            velocity_dw[-1] = momentum_const * velocity_dw[-1] - np.dot(
                delta, activations[-2].transpose()
            ) * (learn_rate / batch_num)

            for i in range(2, self.num_layers):
                delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_prime(
                    zs[-i]
                )
                velocity_db[-i] = momentum_const * velocity_db[-i] - np.sum(
                    delta, axis=1
                ).reshape((-1, 1)) * (learn_rate / batch_num)
                velocity_dw[-i] = momentum_const * velocity_dw[-i] - np.dot(
                    delta, activations[-i - 1].transpose()
                ) * (learn_rate / batch_num)

            self.weights = [w + dw for w, dw in zip(self.weights, velocity_dw)]
            self.bias = [b + db for b, db in zip(self.bias, velocity_db)]
            print(
                f"Epoch {ep} -> Cost {np.sum((activations[-1] - training_y) ** 2) * 0.5/len(training_x)}"
            )

    def print_parameters(self):
        print("\t-----Weights------\t")
        print(self.weights)
        print("\t-----Bias------\t")
        print(self.bias)


if __name__ == "__main__":
    training_x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    training_y = np.array([0, 1, 1, 0])
    m = Network(np.array([2, 2, 1]))
    # training_x: np.ndarray, training_y: np.ndarray, epoch: int, learn_rate: float, momentum_const: float,
    m.train(training_x, training_y, 10000, 0.1, 0.9)
    for x in training_x:
        x = np.resize(x, new_shape=(2, 1))
        res = m.feed_forward(x)
        print(f"{res.round().flatten()} for {x.flatten()}")
