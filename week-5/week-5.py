import numpy as np


class Network_1:
    def __init__(self, weights, learning_rate=0.01):
        self.weights = weights
        self.learning_rate = learning_rate
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def Linear(self, x):
        return x

    def MSE(self, y_true, y_pred):
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

    def zero_grad(self):
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }

    def compute_layer(self, inputs, weights, activation="relu"):
        inputs = np.array(inputs)
        output = []
        weighted_sums = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            weighted_sums.append(weighted_sum)
            output.append(
                self.ReLU(weighted_sum)
                if activation == "relu"
                else self.Linear(weighted_sum)
            )
        return np.array(output), np.array(weighted_sums)

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.hidden_layer, self.hidden_sums = self.compute_layer(
            self.inputs, self.weights[1], activation="relu"
        )
        self.second_hidden_layer, self.second_hidden_sums = self.compute_layer(
            self.hidden_layer.flatten(), self.weights[2], activation="relu"
        )
        self.output_layer, self.output_sums = self.compute_layer(
            self.second_hidden_layer, self.weights[3], activation="linear"
        )
        return self.output_layer

    def backward(self, expected):
        expected = np.array(expected)
        output_errors = self.output_layer - expected
        second_hidden_errors = np.dot(
            output_errors, np.array(self.weights[3])[:, :-1]
        ) * self.ReLU_derivative(self.second_hidden_sums)
        hidden_errors = np.dot(
            second_hidden_errors, np.array(self.weights[2])[:, :-1]
        ) * self.ReLU_derivative(self.hidden_sums)

        for i in range(len(self.weights[3])):
            for j in range(len(self.weights[3][i]) - 1):
                self.gradients[3][i][j] = output_errors[i] * self.second_hidden_layer[j]
            self.gradients[3][i][-1] = output_errors[i]

        for i in range(len(self.weights[2])):
            for j in range(len(self.weights[2][i]) - 1):
                self.gradients[2][i][j] = second_hidden_errors[i] * self.hidden_layer[j]
            self.gradients[2][i][-1] = second_hidden_errors[i]

        for i in range(len(self.weights[1])):
            for j in range(len(self.weights[1][i]) - 1):
                self.gradients[1][i][j] = hidden_errors[i] * self.inputs[j]
            self.gradients[1][i][-1] = hidden_errors[i]

    def update_weights(self):
        for layer in self.weights.keys():
            self.weights[layer] -= self.learning_rate * self.gradients[layer]

    def train(self, test_cases, epochs=1, task_name="", print_last_only=False):
        print(f"--------------- {task_name} ----------------")
        for epoch in range(epochs):
            for case in test_cases:
                self.zero_grad()
                outputs = self.forward(case["input"])
                loss = self.MSE(case["expected"], outputs)
                self.backward(case["expected"])
                self.update_weights()

            if not print_last_only or (epoch + 1 == epochs):
                print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss:.6f}")

        for layer, w in self.weights.items():
            print(f"Layer {layer}:")
            for neuron in w:
                print("  ", neuron)
        print("\n")


class Network_2:
    def __init__(self, weights, learning_rate=0.1):
        self.weights = weights
        self.learning_rate = learning_rate
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Sigmoid_derivative(self, x):
        return self.Sigmoid(x) * (1 - self.Sigmoid(x))

    def BCE(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def BCE_derivative(self, y_true, y_pred):
        return y_pred - y_true

    def zero_grad(self):
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }

    def compute_layer(self, inputs, weights, activation="relu"):
        inputs = np.array(inputs)
        output = []
        weighted_sums = []
        for neuron_weights in weights:
            weighted_sum = np.dot(inputs, neuron_weights[:-1]) + neuron_weights[-1]
            weighted_sums.append(weighted_sum)
            output.append(
                self.ReLU(weighted_sum)
                if activation == "relu"
                else self.Sigmoid(weighted_sum)
            )
        return np.array(output), np.array(weighted_sums)

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.hidden_layer, self.hidden_sums = self.compute_layer(
            self.inputs, self.weights[1], activation="relu"
        )
        self.output_layer, self.output_sums = self.compute_layer(
            self.hidden_layer, self.weights[2], activation="sigmoid"
        )
        return self.output_layer

    def backward(self, expected):
        expected = np.array(expected)
        output_errors = self.output_layer - expected
        hidden_errors = np.dot(
            output_errors, np.array(self.weights[2])[:, :-1]
        ) * self.ReLU_derivative(self.output_sums)

        for i in range(len(self.weights[2])):
            for j in range(len(self.weights[2][i]) - 1):
                self.gradients[2][i][j] = hidden_errors[i] * self.hidden_layer[j]
            self.gradients[2][i][-1] = hidden_errors[i]

        for i in range(len(self.weights[1])):
            self.gradients[1][i][-1] = hidden_errors[i]

        for i in range(len(self.weights[1])):
            for j in range(len(self.weights[1][i]) - 1):
                self.gradients[1][i][j] = hidden_errors[i] * self.inputs[j]
            self.gradients[1][i][-1] = hidden_errors[i]

    def update_weights(self):
        for layer in self.weights.keys():
            self.weights[layer] -= self.learning_rate * self.gradients[layer]

    def train(self, test_cases, epochs=1, task_name="", print_last_only=False):
        print(f"--------------- {task_name} ----------------")
        for epoch in range(epochs):
            for case in test_cases:
                self.zero_grad()
                outputs = self.forward(case["input"])
                loss = self.BCE(case["expected"], outputs)
                self.backward(case["expected"])
                self.update_weights()

            if not print_last_only or (epoch + 1 == epochs):
                print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss:.6f}")

        for layer, w in self.weights.items():
            print(f"Layer {layer}:")
            for neuron in w:
                print("  ", neuron)
        print("\n")


weights_task_1 = {
    1: np.array([[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]]),
    2: np.array([[0.8, -0.5, 0.6]]),
    3: np.array([[0.6, 0.4], [-0.3, 0.75]]),
}

weights_task_2 = {
    1: np.array([[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]]),
    2: np.array([[0.8, 0.4, -0.5]]),
}


nn1 = Network_1(weights_task_1, learning_rate=0.01)
nn2 = Network_2(weights_task_2, learning_rate=0.1)


test_cases_1 = [
    {"input": [1.5, 0.5], "expected": [0.8, 1]},
]

test_cases_2 = [
    {"input": [0.75, 1.25], "expected": [1]},
]

print("--------------- Model 1 ----------------")
nn1.train(test_cases_1, epochs=1, task_name="Task 1")
nn1.train(test_cases_1, epochs=1000, task_name="Task 2", print_last_only=True)

print("--------------- Model 2 ----------------")
nn2.train(test_cases_2, epochs=1, task_name="Task 1")
nn2.train(test_cases_2, epochs=1000, task_name="Task 2", print_last_only=True)
