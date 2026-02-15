"""
model.py
"""
import numpy as np

def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return x * w + b

def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    m = len(x)
    y_hat = predict(x, w, b)
    error = y_hat - y
    return (1 / (2 * m)) * np.sum(error ** 2)

def gradient_descent(x: np.ndarray, y: np.ndarray, w: float, b: float, alpha: float, num_iters: int) -> tuple [float, float, list]:
    m = len(x)
    cost_history = []
    for i in range(num_iters):
        y_hat = predict(x, w, b)
        error = y_hat - y
        dw = (1 / m) * np.sum(x * error)
        db = (1 / m) * np.sum(error)
        w -= alpha * dw
        b -= alpha * db

        cost_history.append(compute_cost(x, y, w, b))
    return w, b, cost_history