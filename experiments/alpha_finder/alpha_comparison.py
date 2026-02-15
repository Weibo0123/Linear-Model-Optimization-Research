"""
alpha_comparison.py
"""
from src.model import gradient_descent
import numpy as np
import csv
import json
import os

data_file = "./data/study_hour_score.csv"
results = []
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
    alpha_list = [0.001, 0.01, 0.1, 1]
    for alpha in alpha_list:
        w0, b0 = 1.0, 1.0
        w, b, cost_history = gradient_descent(x, y, w0, b0, alpha, 100)

        final_cost = cost_history[-1] if len(cost_history) > 0 else None
        status = "stable"
        if final_cost is None or not np.isfinite(final_cost):
            status = "diverged"
        elif len(cost_history) > 2 and cost_history[-1] > cost_history[-2]:
            status = "unstable"

        results.append({
            "alpha": alpha,
            "cost": final_cost,
            "iterations": len(cost_history),
            "status": status
        })

        os.makedirs("./experiments/alpha_finder/results", exist_ok=True)

        with open("./experiments/alpha_finder/results/results.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"alpha: {alpha}, cost: {cost_history[-1]}")

if __name__ == "__main__":
    main()
