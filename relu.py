import numpy as np

from utils import relu, relu_prime


class Network:
    def __init__(self, nodes_num: np.ndarray, learn_rate):
        self.learn_rate = learn_rate
        self.num_layers = len(nodes_num)

        mean = 0
        input_node_length = nodes_num[0]
        std = np.sqrt(2 / input_node_length)

        self.weights: list[np.ndarray] = [
            np.random.normal(mean, std, (nodes_num[i + 1], nodes_num[i]))
            for i in range(len(nodes_num) - 1)
        ]
        self.bias: list[np.ndarray] = [
            np.zeros((nodes_num[i], 1)) for i in range(1, len(nodes_num))
        ]

    def feed_forward(self, input: np.ndarray):
        for w, b in zip(self.weights, self.bias):
            input = relu(np.dot(w, input) + b)
        return input

    def cost_function(self, input: np.ndarray, y):
        res = self.feed_forward(input)
        return ((y - res) ** 2) * 0.5

    def train(self, training_x: np.ndarray, training_y: np.ndarray, epoch: int):
        for ep in range(epoch):
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.bias]

            total_cost = 0
            for x, y in zip(training_x, training_y):
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.bias]

                z = x
                zs = []
                activations = [x]

                for w, b in zip(self.weights, self.bias):
                    z = np.dot(w, z) + b
                    zs.append(z)
                    s = relu(z)
                    activations.append(s)

                delta = (activations[-1] - y) * relu_prime(zs[-1])
                nabla_b[-1] = delta
                nabla_w[-1] = np.dot(delta, activations[-2].transpose())

                for i in range(2, self.num_layers):
                    delta = np.dot(
                        self.weights[-i + 1].transpose(), delta
                    ) * relu_prime(zs[-i])
                    nabla_b[-i] = delta
                    nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())

                delta_w = [dw + nw for dw, nw in zip(delta_w, nabla_w)]
                delta_b = [db + nb for db, nb in zip(delta_b, nabla_b)]

                total_cost += np.sum((self.feed_forward(x) - y) ** 2) / 2

            self.weights = [
                w - ((self.learn_rate / len(training_x)) * dw)
                for w, dw in zip(self.weights, delta_w)
            ]
            self.bias = [
                b - ((self.learn_rate / len(training_x)) * db)
                for b, db in zip(self.bias, delta_b)
            ]
            print(f"Epoch {ep} -> Cost {total_cost/len(training_x)}")

    def print_parameters(self):
        print("\t-----Weights------\t")
        print(self.weights)
        print("\t-----Bias------\t")
        print(self.bias)


if __name__ == "__main__":
    training_x = np.array([[[1], [1]], [[1], [0]], [[0], [1]], [[0], [0]]])
    training_y = np.array([1, 0, 0, 0])
    m = Network(np.array([2, 2, 1]), 0.1)
    m.train(training_x, training_y, 10000)
    m.print_parameters()
    for x in training_x:
        res = m.feed_forward(x)
        print(f"{res} for {x.flatten()}")
