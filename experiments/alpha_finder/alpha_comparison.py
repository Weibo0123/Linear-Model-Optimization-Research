"""
alpha_comparison.py
"""
import numpy as np
import csv
from src.model import gradient_descent

data_file = "./data/study_hour_score.csv"
output_file = "output.csv"
def main():
    hours = []
    scores = []
    with open(data_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hours.append(float(row["Hours"]))
            scores.append(float(row["Scores"]))

    x = np.array(hours)
    y = np.array(scores)
    w = 1
    b = 1
    alpha_list = [0.001, 0.01, 0.1, 1]
    for alpha in alpha_list:
        w0, b0 = 1.0, 1.0
        w, b, cost_history = gradient_descent(x, y, w0, b0, alpha, 100)
        print(f"alpha: {alpha}, cost: {cost_history[-1]}")

if __name__ == "__main__":
    main()
