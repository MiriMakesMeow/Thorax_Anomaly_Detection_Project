import numpy as np

def relu(x: float) -> float:
    if x > 0.0:
        return min(x, 1.0)
    else:
        return 0.0


def drelu(last_output: float) -> float:
    if last_output > 0.0:
        dodz = 1.0
    else:
        dodz = 0.0
    return dodz


def lrelu(x: float) -> float:
    if x > 0.0:
        return min(x, 1.0)
    else:
        return max(0.1 * x, 0.0)


def dlrelu(last_output: float) -> float:
    if last_output > 0.0:
        dodz = 1.0
    else:
        dodz = 0.1
    return dodz


def sigmoid(x: float) -> float:
    return (1.0 / (1 + np.exp(-x)))


def dsigmoid(last_output: float) -> float:
    dodz = last_output * (1.0 - last_output)
    return dodz


def get_activation_and_derivative_and_name(activation_function: str):
    activation_function = activation_function.lower()
    if activation_function == "sigmoid":
        return sigmoid, dsigmoid, activation_function
    elif activation_function == "relu":
        return relu, drelu, activation_function
    elif activation_function == "lrelu":
        return lrelu, dlrelu, activation_function
    else:
        raise Exception("Unknown activation", activation_function)
