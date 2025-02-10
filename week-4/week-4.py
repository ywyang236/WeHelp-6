import numpy as np


class Network_1:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def ReLU(self, x):
        return np.maximum(0, x)

    def Linear(self, x):
        return x

    def MSE(self, y_true, y_pred):
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

    def compute_layer(self, inputs, weights, activation="relu"):
        output = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            output.append(
                self.ReLU(weighted_sum)
                if activation == "relu"
                else self.Linear(weighted_sum)
            )
        return np.array(output)

    def forward(self, inputs, expected):
        hidden_layer = self.compute_layer(inputs, self.weights[1], activation="relu")
        output = self.compute_layer(hidden_layer, self.weights[2], activation="linear")
        output = np.array(output)
        loss = self.MSE(np.array(expected), output)
        return output, loss

    def test(self, test_cases):
        for case in test_cases:
            output, loss = self.forward(
                np.array(case["input"]), np.array(case["expected"])
            )
            print(f"Total Loss: {loss}")


class Network_2:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def ReLU(self, x):
        return np.maximum(0, x)

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def BinaryCrossEntropy(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_layer(self, inputs, weights, activation="relu"):
        output = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            output.append(
                self.ReLU(weighted_sum)
                if activation == "relu"
                else self.Sigmoid(weighted_sum)
            )
        return np.array(output)

    def forward(self, inputs, expected):
        hidden_layer = self.compute_layer(inputs, self.weights[1], activation="relu")
        output = self.compute_layer(hidden_layer, self.weights[2], activation="sigmoid")
        output = np.array(output)
        loss = self.BinaryCrossEntropy(expected, output)
        return output, loss

    def test(self, test_cases):
        for case in test_cases:
            output, loss = self.forward(
                np.array(case["input"]), np.array(case["expected"])
            )
            print(f"Total Loss: {loss}")


class Network_3:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def ReLU(self, x):
        return np.maximum(0, x)

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def BinaryCrossEntropy(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_layer(self, inputs, weights, activation="relu"):
        output = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            output.append(
                self.ReLU(weighted_sum)
                if activation == "relu"
                else self.Sigmoid(weighted_sum)
            )
        return np.array(output)

    def forward(self, inputs, expected):
        hidden_layer = self.compute_layer(inputs, self.weights[1], activation="relu")
        output = self.compute_layer(hidden_layer, self.weights[2], activation="sigmoid")
        output = np.array(output)
        loss = self.BinaryCrossEntropy(expected, output)
        return output, loss

    def test(self, test_cases):
        for case in test_cases:
            output, loss = self.forward(
                np.array(case["input"]), np.array(case["expected"])
            )
            print(f"Total Loss: {loss}")


class Network_4:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def ReLU(self, x):
        return np.maximum(0, x)

    def Softmax(self, x):
        x = np.atleast_2d(x)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def CategoricalCrossEntropy(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(1, -1)
        y_pred = np.array(y_pred).reshape(1, -1)

        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    def compute_layer(self, inputs, weights, activation="relu"):
        output = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            output.append(weighted_sum)

        output = np.array(output).reshape(1, -1)

        if activation == "relu":
            return self.ReLU(output)
        elif activation == "softmax":
            return self.Softmax(output)
        else:
            return output

    def forward(self, inputs, expected):
        hidden_layer = self.compute_layer(inputs, self.weights[1], activation="relu")
        output = self.compute_layer(hidden_layer, self.weights[2], activation="softmax")
        loss = self.CategoricalCrossEntropy(expected, output)
        return output, loss

    def test(self, test_cases):
        for case in test_cases:
            output, loss = self.forward(
                np.array(case["input"]), np.array(case["expected"])
            )
            print(f"Total Loss: {loss}")


weights_task_1 = {
    1: [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],
    2: [[0.8, -0.5, 0.6], [0.4, 0.5, -0.25]],
}

weights_task_2 = {
    1: [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],
    2: [[0.8, 0.4, -0.5]],
}

weights_task_3 = {
    1: [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],
    2: [[0.8, -0.4, 0.6], [0.5, 0.4, 0.5], [0.3, 0.75, -0.5]],
}

weights_task_4 = {
    1: [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],
    2: [[0.8, -0.4, 0.6], [0.5, 0.4, 0.5], [0.3, 0.75, -0.5]],
}


bias = 1

nn1 = Network_1(weights_task_1, bias)
nn2 = Network_2(weights_task_2, bias)
nn3 = Network_3(weights_task_3, bias)
nn4 = Network_4(weights_task_4, bias)


test_cases_1 = [
    {"input": [1.5, 0.5], "expected": [0.8, 1]},
    {"input": [0, 1], "expected": [0.5, 0.5]},
]

test_cases_2 = [
    {"input": [0.75, 1.25], "expected": [1]},
    {"input": [-1, 0.5], "expected": [0]},
]

test_cases_3 = [
    {"input": [1.5, 0.5], "expected": [1, 0, 1]},
    {"input": [0, 1], "expected": [1, 1, 0]},
]

test_cases_4 = [
    {"input": [1.5, 0.5], "expected": [1, 0, 0]},
    {"input": [0, 1], "expected": [0, 0, 1]},
]

nn1.test(test_cases_1)
print("-" * 40)
nn2.test(test_cases_2)
print("-" * 40)
nn3.test(test_cases_3)
print("-" * 40)
nn4.test(test_cases_4)
