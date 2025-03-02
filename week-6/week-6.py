import csv
import os
import numpy as np
import torch


class Network_1:
    def __init__(self, input_size, hidden_size=4, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.weights = {
            1: np.random.randn(input_size, hidden_size) * 0.01,
            2: np.random.randn(hidden_size, 1) * 0.01,
        }
        self.biases = {
            1: np.zeros((1, hidden_size)),
            2: np.zeros((1, 1)),
        }
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }
        self.bias_gradients = {
            layer: np.zeros_like(bias) for layer, bias in self.biases.items()
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
        self.bias_gradients = {
            layer: np.zeros_like(bias) for layer, bias in self.biases.items()
        }

    def forward(self, X):
        self.layer1_input = X

        self.layer1_weighted_sum = np.dot(X, self.weights[1]) + self.biases[1]
        self.layer1_activation = self.ReLU(self.layer1_weighted_sum)

        self.layer2_weighted_sum = (
            np.dot(self.layer1_activation, self.weights[2]) + self.biases[2]
        )
        self.layer2_output = self.Linear(self.layer2_weighted_sum)

        return self.layer2_output

    def backward(self, X, y):
        m = X.shape[0]

        layer2_error = self.layer2_output - y
        self.gradients[2] = np.dot(self.layer1_activation.T, layer2_error) / m
        self.bias_gradients[2] = np.sum(layer2_error, axis=0, keepdims=True) / m

        layer1_error = np.dot(layer2_error, self.weights[2].T)
        layer1_delta = layer1_error * self.ReLU_derivative(self.layer1_weighted_sum)
        self.gradients[1] = np.dot(X.T, layer1_delta) / m
        self.bias_gradients[1] = np.sum(layer1_delta, axis=0, keepdims=True) / m

    def update_weights(self):
        for layer in self.weights.keys():
            self.weights[layer] -= self.learning_rate * self.gradients[layer]
            self.biases[layer] -= self.learning_rate * self.bias_gradients[layer]

    def train(self, X, y, epochs=1000, batch_size=32, print_last_only=False):
        m = X.shape[0]
        print("Start training model...")

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                self.zero_grad()
                output = self.forward(X_batch)
                loss = self.MSE(y_batch, output)
                self.backward(X_batch, y_batch)
                self.update_weights()

            if (epoch + 1) % 250 == 0:
                predictions = self.forward(X)
                loss = self.MSE(y, predictions)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

    @staticmethod
    def load_data(file_path):
        data = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "gender-height-weight.csv")

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                data.append(row)

        print(f"Loaded {len(data)} samples from {file_path}")
        return header, data

    @staticmethod
    def preprocess_data(data):
        input_features = []
        target_values = []

        for row in data:
            gender = 1 if row[0] == "Male" else 0
            height = float(row[1])
            weight = float(row[2])
            input_features.append([gender, height])
            target_values.append(weight)

        input_features = np.array(input_features)
        target_values = np.array(target_values).reshape(-1, 1)

        height_mean = np.mean(input_features[:, 1])
        height_std = np.std(input_features[:, 1])
        weight_mean = np.mean(target_values)
        weight_std = np.std(target_values)

        input_features[:, 1] = (input_features[:, 1] - height_mean) / height_std
        target_values = (target_values - weight_mean) / weight_std

        return input_features, target_values, weight_mean, weight_std


class Network_2:
    def __init__(self, input_size, hidden_size=4, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.weights = {
            1: np.random.randn(input_size, hidden_size) * 0.01,
            2: np.random.randn(hidden_size, 1) * 0.01,
        }
        self.biases = {1: np.zeros((1, hidden_size)), 2: np.zeros((1, 1))}

        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }
        self.bias_gradients = {
            layer: np.zeros_like(bias) for layer, bias in self.biases.items()
        }

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def BCE_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward(self, X):
        self.layer1_input = X

        self.layer1_weighted_sum = np.dot(X, self.weights[1]) + self.biases[1]
        self.layer1_activation = self.ReLU(self.layer1_weighted_sum)

        self.layer2_weighted_sum = (
            np.dot(self.layer1_activation, self.weights[2]) + self.biases[2]
        )
        self.layer2_output = self.sigmoid(self.layer2_weighted_sum)

        return self.layer2_output

    def backward(self, X, y):
        m = X.shape[0]

        layer2_error = self.layer2_output - y
        self.gradients[2] = np.dot(self.layer1_activation.T, layer2_error) / m
        self.bias_gradients[2] = np.sum(layer2_error, axis=0, keepdims=True) / m

        layer1_error = np.dot(layer2_error, self.weights[2].T)
        layer1_delta = layer1_error * self.ReLU_derivative(self.layer1_weighted_sum)
        self.gradients[1] = np.dot(X.T, layer1_delta) / m
        self.bias_gradients[1] = np.sum(layer1_delta, axis=0, keepdims=True) / m

    def update_weights(self):
        for layer in self.weights.keys():
            self.weights[layer] -= self.learning_rate * self.gradients[layer]
            self.biases[layer] -= self.learning_rate * self.bias_gradients[layer]

    def zero_grad(self):
        self.gradients = {
            layer: np.zeros_like(weights) for layer, weights in self.weights.items()
        }
        self.bias_gradients = {
            layer: np.zeros_like(bias) for layer, bias in self.biases.items()
        }

    @staticmethod
    def load_data(file_path):
        data = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                data.append(row)

        print(f"Loaded {len(data)} samples from {file_path}")
        return header, data

    @staticmethod
    def preprocess_data(data, show_head=False):
        data = np.array(data)

        pclass = data[:, 2].astype(float)
        sex = data[:, 4]

        age = np.array([float(x) if x != "" else np.nan for x in data[:, 5]])

        sibsp = data[:, 6].astype(float)
        parch = data[:, 7].astype(float)

        fare = np.array([float(x) if x != "" else np.nan for x in data[:, 9]])

        cabin = data[:, 10]
        embarked = data[:, 11]
        names = data[:, 3]

        age_mean = np.nanmean(age)
        fare_mean = np.nanmean(fare)

        age[np.isnan(age)] = age_mean
        fare[np.isnan(fare)] = fare_mean

        pclass_1 = (pclass == 1).astype(float)
        pclass_2 = (pclass == 2).astype(float)
        pclass_3 = (pclass == 3).astype(float)

        sex_male = (sex == "male").astype(float)
        sex_female = (sex == "female").astype(float)

        is_master = np.array(["Master." in name for name in names]).astype(float)

        has_cabin = (cabin != "").astype(float)

        embarked_S = (embarked == "S").astype(float)
        embarked_C = (embarked == "C").astype(float)
        embarked_Q = (embarked == "Q").astype(float)

        age = (age - np.mean(age)) / np.std(age)
        fare = (fare - np.mean(fare)) / np.std(fare)
        sibsp = (sibsp - np.mean(sibsp)) / np.std(sibsp)
        parch = (parch - np.mean(parch)) / np.std(parch)

        X = np.column_stack(
            (
                pclass_1,
                pclass_2,
                pclass_3,
                sex_male,
                sex_female,
                age,
                sibsp,
                parch,
                fare,
                has_cabin,
                embarked_S,
                embarked_C,
                embarked_Q,
                is_master,
            )
        )

        y = data[:, 1].astype(float).reshape(-1, 1)

        return X, y

    def train(self, X, y, epochs=1000, batch_size=32, print_last_only=False):
        m = X.shape[0]
        print("Start training model...")

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                self.zero_grad()
                output = self.forward(X_batch)
                loss = self.BCE_loss(y_batch, output)
                self.backward(X_batch, y_batch)
                self.update_weights()

            if (epoch + 1) % 250 == 0:
                predictions = self.forward(X)
                loss = self.BCE_loss(y, predictions)
                accuracy = np.mean((predictions >= 0.5) == y)
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                )

    def predict(self, X):
        return self.forward(X) >= 0.5


def task_1():
    header, raw_data = Network_1.load_data("gender-height-weight.csv")
    X, y, weight_mean, weight_std = Network_1.preprocess_data(raw_data)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.8)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    model = Network_1(input_size=2, hidden_size=3, learning_rate=0.01)
    model.train(X_train, y_train, epochs=1000, print_last_only=False)

    y_pred = model.predict(X_test)
    z_score_error = np.mean((y_pred - y_test) ** 2)

    y_pred_original = y_pred * weight_std + weight_mean
    y_test_original = y_test * weight_std + weight_mean

    pound_error = np.mean(np.abs(y_pred_original - y_test_original))
    mse_pounds = np.mean((y_pred_original - y_test_original) ** 2)
    percentage_error = np.mean(
        np.abs(y_pred_original - y_test_original) / y_test_original * 100
    )

    print(f"Average Test Loss (z-score): {z_score_error:.6f}")
    print(f"Mean Squared Error (pounds): {mse_pounds:.2f}")
    print(f"Mean Absolute Error (average pounds off): {pound_error:.2f}")
    print(f"Average Percentage Error: {percentage_error:.2f}%")


def task_2():
    header, raw_data = Network_2.load_data("titanic.csv")
    X, y = Network_2.preprocess_data(raw_data)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.8)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    input_size = X.shape[1]

    model = Network_2(input_size=input_size, hidden_size=8, learning_rate=0.1)

    model.train(X_train, y_train, epochs=1000, print_last_only=True)

    train_predictions = model.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train)

    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Average Training Accuracy (survival): {train_accuracy*100:.2f} %")
    print(f"Average Test Accuracy (survival): {test_accuracy*100:.2f} %")


def task_3():
    tensor1 = torch.tensor([[2, 3, 1], [5, -2, 1]])
    print(f"\n----------------- Task 3-1 ------------------")
    print("Shape:", tensor1.shape)
    print("Dtype:", tensor1.dtype)
    print(tensor1)
    print()

    tensor2 = torch.rand(3, 4, 2)
    print(f"\n----------------- Task 3-2 ------------------")
    print("Shape:", tensor2.shape)
    print(tensor2)
    print()

    tensor3 = torch.ones(2, 1, 5)
    print(f"\n----------------- Task 3-3 ------------------")
    print("Shape:", tensor3.shape)
    print(tensor3)
    print()

    tensor4a = torch.tensor([[1, 2, 4], [2, 1, 3]])
    tensor4b = torch.tensor([[5], [2], [1]])
    result4 = torch.matmul(tensor4a, tensor4b)
    print(f"\n----------------- Task 3-4 ------------------")
    print(result4)
    print()

    tensor5a = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    tensor5b = torch.tensor([[5, 4], [2, 1], [1, -5]])
    result5 = tensor5a * tensor5b
    print(f"\n----------------- Task 3-5 ------------------")
    print(result5)


if __name__ == "__main__":
    print(f"------------------ Task 1 -------------------")
    task_1()
    print("-" * 45)

    print(f"\n------------------ Task 2 -------------------")
    task_2()
    print("-" * 45)

    task_3()
    print("-" * 45)
