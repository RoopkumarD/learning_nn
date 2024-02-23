import pickle

import numpy as np

from utils import sigmoid, sigmoid_prime, softmax, softmax_prime

np.random.seed(42)


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
        for i in range(self.num_layers - 2):
            z = sigmoid(np.dot(self.weights[i], z) + self.bias[i])
        z = softmax(np.dot(self.weights[-1], z) + self.bias[-1])
        return z

    def one_hot_encoding(self, train_y: np.ndarray):
        one_hot = np.zeros((len(train_y), 10))
        one_hot[np.arange(train_y.size), train_y] = 1
        return one_hot

    def train(self, train_data: dict, epoch: int, learn_rate: float, batch_num: int):
        length_of_train = len(train_data["y"])
        train_data_x = train_data["x"]
        train_data_y = self.one_hot_encoding(train_data["y"])
        # normalising it so that it doesn't explode exponential
        train_data_x = train_data_x / 255
        # I seem to face numerical instability, so i am normalising the input
        train_data_x = (train_data_x - train_data_x.mean()) / train_data_x.std()
        arr = np.array(range(length_of_train))

        for ep in range(epoch):
            print(f"Epoch {ep}")
            np.random.shuffle(arr)

            batches = [
                (
                    train_data_x[arr[i : i + batch_num]],
                    train_data_y[arr[i : i + batch_num]],
                )
                for i in range(0, length_of_train, batch_num)
            ]

            for x, y in batches:
                delta_w = [np.zeros(w.shape) for w in self.weights]
                delta_b = [np.zeros(b.shape) for b in self.bias]

                z = x.transpose()
                zs = []
                activations = [np.array(z)]

                for q in range(self.num_layers - 2):
                    z = np.dot(self.weights[q], z) + self.bias[q]
                    zs.append(z)
                    z = sigmoid(z)
                    activations.append(z)

                z = np.dot(self.weights[-1], z) + self.bias[-1]
                zs.append(z)
                z = softmax(z)
                activations.append(z)

                diff = activations[-1] - y.T
                softmax_grad = softmax_prime(activations[-1])
                delta = np.einsum("kij,jk->ik", softmax_grad, diff)
                delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
                delta_w[-1] = np.dot(delta, activations[-2].transpose())

                for i in range(2, self.num_layers):
                    delta = np.dot(
                        self.weights[-i + 1].transpose(), delta
                    ) * sigmoid_prime(zs[-i])
                    delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
                    delta_w[-i] = np.dot(delta, activations[-i - 1].transpose())

                self.weights = [
                    w - ((learn_rate / batch_num) * dw)
                    for w, dw in zip(self.weights, delta_w)
                ]
                self.bias = [
                    b - ((learn_rate / batch_num) * db)
                    for b, db in zip(self.bias, delta_b)
                ]

            pred = self.feed_forward(train_data_x.T)
            print(
                f"Epoch {ep} -> Cost {np.sum((pred - train_data_y.T) ** 2) / (2 * length_of_train)}"
            )

    def print_parameters(self):
        print("\t-----Weights------\t")
        print(self.weights)
        print("\t-----Bias------\t")
        print(self.bias)


if __name__ == "__main__":
    print("Loading training data")
    with open("./mnist_train.pkl", "rb") as train_file:
        train_data = pickle.load(train_file)
    m = Network(np.array([784, 30, 10]))
    print("Training network")
    # train_data: dict, epoch: int, learn_rate: float, batch_num: int
    m.train(train_data, 30, 3, 10)

    # saving weights and bias
    with open("sigmoid_wb.pkl", "wb") as f:
        o = {"weights": m.weights, "bias": m.bias}
        pickle.dump(o, f)

    print("Checking for accuracy against test")
    with open("./mnist_test.pkl", "rb") as test_file:
        test_data = pickle.load(test_file)

    x, y = test_data["x"], test_data["y"]
    x = x.transpose()
    x = x / 255
    pred = m.feed_forward(x)
    pred = np.argmax(pred.T, axis=1)
    length_of_test = len(y)
    total_right = np.sum(((y - pred) == 0) * 1)

    print(f"Gave correct results for {(total_right/length_of_test) * 100}")
