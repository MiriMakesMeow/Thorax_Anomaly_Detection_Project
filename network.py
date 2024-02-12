import numpy as np
from layers import InputLayer, FullyConnectedMatrixLayer, OutputLayer
from error_metric import get_error_and_derivative_and_name

class NetworkMatrix():
    def __init__(self, number_inputs: int, number_outputs: int, number_hidden: int = 100, error_metric: str = "bc"):
        self.layers = []
        self.layers.append(InputLayer(number_inputs=number_inputs))
        self.layers.append(FullyConnectedMatrixLayer(number_inputs=number_inputs, number_hidden=number_hidden))
        self.layers.append(OutputLayer(number_hidden=number_hidden, number_outputs=number_outputs))
        self.error_metric, self.error_metric_deriv, _ = get_error_and_derivative_and_name(error_metric)

    def train(self, inputs, targets, learning_rate: float = 0.02):
        prediction = self.query(inputs)
        loss = self.error_metric(prediction, targets)
        # error metric
        next_errors = self.error_metric_deriv(prediction, targets)

        for layer in reversed(self.layers):
            next_errors = layer.train(next_errors, learning_rate=learning_rate)
        return loss

    def query(self, next_input):
        for layer in self.layers:
            next_input = layer.query(next_input)
        return next_input

    def print(self):
        print("A Neural Network with", len(self.layers), " Layers")
        for layer in self.layers:
            layer.print()
