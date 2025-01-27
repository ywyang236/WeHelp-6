class Network:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def linear(self, x):
        return x

    def compute_layer(self, inputs, weights):
        outputs = []
        for neuron_weights in weights:
            output = (
                sum(w * x for w, x in zip(neuron_weights[:-1], inputs))
                + neuron_weights[-1] * self.bias
            )
            outputs.append(self.linear(output))
        return outputs

    def forward(self, inputs):
        layer_inputs = list(inputs)
        for layer in sorted(self.weights.keys()):
            layer_inputs = self.compute_layer(layer_inputs, self.weights[layer])
        return layer_inputs


weights_task_1 = {
    1: [
        [0.5, 0.2, 0.3],
        [0.6, -0.6, 0.25],
    ],
    2: [
        [0.8, 0.4, -0.5],
    ],
}

weights_task_2 = {
    1: [
        [0.5, 1.5, 0.3],
        [0.6, -0.8, 1.25],
    ],
    2: [
        [0.6, -0.8, 0.3],
    ],
    3: [
        [0.5, 0.2],
        [-0.4, 0.5],
    ],
}

bias = 1

nn = Network(weights_task_1, bias)
inputs_1 = (1.5, 0.5)
outputs = nn.forward(inputs_1)
print(outputs)

inputs_2 = (0, 1)
outputs = nn.forward(inputs_2)
print(outputs)

print("=" * 40)

nn = Network(weights_task_2, bias)
inputs_3 = (0.75, 1.25)
outputs = nn.forward(inputs_3)
print(outputs)

inputs_4 = (-1, 0.5)
outputs = nn.forward(inputs_4)
print(outputs)
