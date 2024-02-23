import matplotlib.pyplot as plt
import numpy as np

from utils import sigmoid, sigmoid_prime


class Network:
    def __init__(self, nodes_num: np.ndarray, learn_rate):
        self.learn_rate = learn_rate
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

    def train(self, training_x: np.ndarray, training_y: np.ndarray, epoch: int):
        # taking every input together
        for ep in range(epoch):
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.bias]

            z = training_x.transpose()
            zs = []
            activations = [np.array(z)]

            for w, b in zip(self.weights, self.bias):
                z = np.dot(w, z) + b
                zs.append(z)
                z = sigmoid(z)
                activations.append(z)

            delta = (activations[-1] - training_y) * sigmoid_prime(zs[-1])
            delta_b[-1] = np.sum(delta, axis=1).reshape((-1, 1))
            delta_w[-1] = np.dot(delta, activations[-2].transpose())

            for i in range(2, self.num_layers):
                delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_prime(
                    zs[-i]
                )
                delta_b[-i] = np.sum(delta, axis=1).reshape((-1, 1))
                delta_w[-i] = np.dot(delta, activations[-i - 1].transpose())

            self.weights = [
                w - ((self.learn_rate / len(training_x)) * dw)
                for w, dw in zip(self.weights, delta_w)
            ]
            self.bias = [
                b - ((self.learn_rate / len(training_x)) * db)
                for b, db in zip(self.bias, delta_b)
            ]
            print(
                f"Epoch {ep} -> Cost {np.sum((activations[-1] - training_y) ** 2) * 0.5/len(training_x)}"
            )

    def print_parameters(self):
        print("\t-----Weights------\t")
        print(self.weights)
        print("\t-----Bias------\t")
        print(self.bias)


def above_line(point):
    return 1.5 - point[0] < point[1]


if __name__ == "__main__":
    training_x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    training_y = np.array([0, 1, 1, 0])
    m = Network(np.array([2, 2, 1]), 1)
    m.train(training_x, training_y, 10000)

    random_points = np.random.randint(-50, 50, size=(100, 2))
    pred = m.feed_forward(random_points.T).round().flatten()

    category_less_than_equal_to_zero = pred <= 0
    category_greater_than_zero = pred > 0

    plt.scatter(
        random_points[category_less_than_equal_to_zero][:, 0],
        random_points[category_less_than_equal_to_zero][:, 1],
        color="blue",
        label="Category <= 0",
    )
    plt.scatter(
        random_points[category_greater_than_zero][:, 0],
        random_points[category_greater_than_zero][:, 1],
        color="red",
        label="Category > 0",
    )

    # Plot the line y = 1.5 - x
    # line_x = np.linspace(min(random_points[:, 0]), max(random_points[:, 0]), 100)
    # line_y = 1.5 - line_x
    # plt.plot(line_x, line_y, color="green", linestyle="--", label="y = 1.5 - x")

    # Customize the plot
    plt.title("Scatter Plot with Categories and Line")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # Save the plot as an image (e.g., PNG)
    # plt.savefig("scatter_plot_with_line.png")

    # Show the plot (optional)
    plt.show()
