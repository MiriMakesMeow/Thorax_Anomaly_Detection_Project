import numpy as np


def mse(last_output: float, target: float) -> float:
    # mean squared error
    return (target - last_output)**2


def dmse(last_output: float, target: float) -> float:
    # derivate of mean squared error w.r.t. last_output
    return target - last_output


def mae(last_output: float, target: float) -> float:
    return np.abs(target - last_output)


def dmae(last_output: float, target: float) -> float:
    return np.sign(target - last_output)

def binary_crossentropy(last_output, target):
    epsilon = 1e-7
    return -np.mean(target * np.log(last_output + epsilon) + (1 - target) * np.log(1 - last_output + epsilon))

def dbinary_crossentropy(last_output, target):
    return -(target / last_output) + (1 - target) / (1 - last_output)


def get_error_and_derivative_and_name(error_metric: str):
    error_metric = error_metric.lower()
    if error_metric == "mae":
        return mae, dmae, error_metric
    elif error_metric == "mse":
        return mse, dmse, error_metric
    elif error_metric == "bc":
        return binary_crossentropy, dbinary_crossentropy, error_metric
    else:
        raise Exception("Unknown error metric", error_metric)
