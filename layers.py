from activation_functions import get_activation_and_derivative_and_name
import numpy as np

class InputLayer():
    def __init__(self, number_inputs) -> None:
        self.number_inputs = number_inputs

    def query(self, x: np.ndarray):
        return x
    
    def train(self, error: float, learning_rate = 0.01):
        pass

    def get_number_outputs(self):
        return self.num_inputs
    
    def print(self):
        print(f'Input Layer with {self.num_inputs} Inputs')

class FullyConnectedMatrixLayer():
    def __init__(self, number_inputs, number_hidden, activation_function: str = "sigmoid"):
        self.w = 2.0 * np.random.random((number_hidden, number_inputs)) - 1.0
        self.activation_function, self.activation_function_deriv, _ = get_activation_and_derivative_and_name(activation_function)

    def query(self, x):
        x = np.array(x, ndmin=2)
        if x.shape[0] != self.w.shape[1]:
            x = x.T
        outputs = self.activation_function(np.dot(self.w, x))
        self.last_outputs = outputs
        self.last_inputs = x
        return outputs

    def train(self, errors, learning_rate: float = 0.02):
        last_outputs = self.last_outputs
        last_inputs = self.last_inputs
        last_errors = np.dot(self.w.T, errors)
        dEdw = np.dot((errors * self.activation_function_deriv(last_outputs)), np.transpose(last_inputs))
        self.w += learning_rate * dEdw
        return last_errors

    def print(self):
        print("Fully connected with ", self.w.shape[0], " Neurons")

class OutputLayer():
    def __init__(self, number_hidden, number_outputs, activation_function: str = "sigmoid"):
        self.w = 2.0 * np.random.random((number_outputs, number_hidden)) - 1.0
        self.activation_function, self.activation_function_deriv, _ = get_activation_and_derivative_and_name(activation_function)

    def query(self, x):
        x = np.array(x, ndmin=2)
        if x.shape[0] != self.w.shape[1]:
            x = x.T
        outputs = self.activation_function(np.dot(self.w, x))
        self.last_outputs = outputs
        self.last_inputs = x
        return outputs

    def train(self, errors, learning_rate: float = 0.02):
        last_outputs = self.last_outputs
        last_inputs = self.last_inputs
        last_errors = np.dot(self.w.T, errors)
        dEdw = np.dot((errors * self.activation_function_deriv(last_outputs)), np.transpose(last_inputs))
        self.w += learning_rate * dEdw
        return last_errors

    def print(self):
        print("Output connected with ", self.w.shape[0], " Neurons")
