"""
Considering Image as Square
"""

import numpy as np

from utils import image_convolution, max_pooling, relu, relu_prime, softmax


class Convolution:
    def __init__(
        self,
        feature_num: int,
        kernel_size: int,
        input_image_size: int,
        output_nodes_num: int,
        stride: int,
        pooling_size: int,
    ) -> None:
        self.feature_weights = [
            np.random.random((kernel_size, kernel_size)) - 0.5
            for _ in range(feature_num)
        ]
        self.kernel_size = kernel_size
        self.feature_bias = np.zeros(feature_num)
        self.stride = stride
        self.output_nodes_num = output_nodes_num
        self.feature_num = feature_num
        self.pooling_size = pooling_size

        feature_map_size = 1 + (input_image_size - kernel_size)
        # considering non overlap pooling
        # will make it general later, for now i want to get handle of forward and backpropogation
        pool_map_size = int(feature_map_size / pooling_size)
        self.pool_map_size = pool_map_size

        self.last_layer_weights = (
            np.random.random(
                (output_nodes_num, pool_map_size * pool_map_size * feature_num)
            )
            - 0.5
        )
        self.last_layer_bias = np.random.random((output_nodes_num, 1)) - 0.5

    def forward_pass(self, image: np.ndarray):
        # first image to convolution layer
        feature_maps = [
            relu(image_convolution(w, image, self.stride) + b)
            for w, b in zip(self.feature_weights, self.feature_bias)
        ]
        # convolution layer (feature map) to pooling layer
        # doing max pooling
        # thus list of (-1, 1) ndarray as did reshaping in max_pooling fn itself
        max_pooled_layer_with_elem = [
            max_pooling(self.pooling_size, feature_maps[i])
            for i in range(self.feature_num)
        ]
        max_pooled_layer, _ = list(zip(*max_pooled_layer_with_elem))
        # lastly combine all pooling layer as such and find ouput
        final_second_last_layer = np.concatenate(max_pooled_layer)
        result = softmax(
            np.dot(self.last_layer_weights, final_second_last_layer)
            + self.last_layer_bias
        )

        return result

    def fit_model(self, train_x, train_y, epoch: int, learning_rate: float):
        train_len = len(train_x)

        for ep in range(epoch):
            results_value = np.zeros((self.output_nodes_num, train_len))

            delta_w_last = np.zeros(self.last_layer_weights.shape)
            delta_b_last = np.zeros(self.last_layer_bias.shape)
            delta_feature_weights = [np.zeros(w.shape) for w in self.feature_weights]
            delta_feature_bias = np.zeros(self.feature_bias.shape)

            for i in range(train_len):
                feature_map_bare = [
                    image_convolution(w, train_x[i], self.stride) + b
                    for w, b in zip(self.feature_weights, self.feature_bias)
                ]
                feature_maps = [relu(f) for f in feature_map_bare]
                # convolution layer (feature map) to pooling layer
                # doing max pooling
                # thus list of (-1, 1) ndarray as did reshaping in max_pooling fn itself
                max_pooled_layer_with_elem = [
                    max_pooling(self.pooling_size, feature_maps[i])
                    for i in range(self.feature_num)
                ]
                max_pooled_layer, pooled_elem = list(zip(*max_pooled_layer_with_elem))
                # lastly combine all pooling layer as such and find ouput
                final_second_last_layer = np.concatenate(max_pooled_layer)
                result = softmax(
                    np.dot(self.last_layer_weights, final_second_last_layer)
                    + self.last_layer_bias
                )
                for ir in range(self.output_nodes_num):
                    results_value[ir][i] = result[ir][0]

                delta = result - train_y[i].reshape((-1, 1))
                delta_b_last += delta
                delta_w_last += np.dot(delta, final_second_last_layer.transpose())
                delta_dash = np.dot(self.last_layer_weights.transpose(), delta)

                for k in range(self.feature_num):
                    temp_delta_dash = delta_dash[k : k + len(max_pooled_layer[k])]
                    temp_delta_dash_len = len(temp_delta_dash)

                    zs = np.zeros(temp_delta_dash_len)
                    ks = np.zeros(
                        (self.kernel_size * self.kernel_size, temp_delta_dash_len)
                    )
                    for m in range(temp_delta_dash_len):
                        r, c = pooled_elem[k][m]
                        zs[m] = feature_map_bare[k][r][c]
                        for ik in range(self.kernel_size):
                            for jk in range(self.kernel_size):
                                ks[ik * self.kernel_size + jk][m] = train_x[i][r + ik][
                                    c + jk
                                ]

                    temp_delta_dash = temp_delta_dash * relu_prime(zs)
                    delta_feature_bias[k] += np.sum(temp_delta_dash)
                    delta_feature_weights[k] += np.sum(
                        np.dot(ks, temp_delta_dash), axis=1, keepdims=True
                    ).reshape((-1, self.kernel_size))

            print(
                f"Epoch {ep} -> Cost {np.sum((results_value - train_y.T) ** 2) / (2 * train_len)}"
            )

            self.last_layer_weights -= delta_w_last * (learning_rate / train_len)
            self.last_layer_bias -= delta_b_last * (learning_rate / train_len)
            self.feature_bias -= delta_feature_bias * (learning_rate / train_len)
            self.feature_weights = [
                w - (dw * (learning_rate / train_len))
                for w, dw in zip(self.feature_weights, delta_feature_weights)
            ]


if __name__ == "__main__":
    x = np.array(
        [
            [
                [0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
            ],
        ]
    )

    # [o, x]
    y = np.array([[1, 0], [0, 1]])

    # feature_num: int, kernel_size: int, input_image_size: int, output_nodes_num: int, stride: int, pooling_size: int
    model = Convolution(1, 3, 6, 2, 1, 2)

    model.fit_model(x, y, 100, 0.5)

    for mx in x:
        value = np.argmax(model.forward_pass(mx))
        m = ""
        if value == 0:
            m = "o"
        else:
            m = "x"
        print(f"{mx} is {m}")
