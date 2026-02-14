"""
model.py
"""
import numpy as np

def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return x * w + b

def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float):
    m = len(x)
    y_hat = predict(x, w, b)
    error = y_hat - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def gradient_descent(x: np.ndarray, y: np.ndarray, w: float, b: float, alpha: float, num_iters: int):
    m = len(x)
    for _ in range(num_iters):
        y_hat = predict(x, w, b)
        dw = (1 / m) * np.sum(x * (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        w -= alpha * dw
        b -= alpha * db
    return w, b