"""
alpha_comparison.py
"""
from src.single_linear_regression import gradient_descent, normal_equation, compute_cost
import numpy as np
import csv
import json
import os

DATA_FILE = "./data/study_hour_score.csv"
alpha_list = [0.001, 0.01, 0.1, 1]
num_iters = 100

def main():
    results = []
    hours, scores, row_skipped = clean_data(DATA_FILE)

    x = np.array(hours)
    y = np.array(scores)

    os.makedirs("./experiments/alpha_finder/results", exist_ok=True)

    w0, b0 = 1.0, 1.0
    meta = {
        "data_file": DATA_FILE,
        "num_rows": len(x),
        "row_skipped": row_skipped,
        "alpha_list": alpha_list,
        "num_iters": num_iters,
        "init": {"w0": w0, "b0": b0},
    }

    w_normal, b_normal = normal_equation(x, y)
    normal_cost = compute_cost(x, y, w_normal, b_normal)

    meta["normal_equation"] = {
        "method": "normal_equation",
        "w": w_normal,
        "b": b_normal,
        "cost": normal_cost
        }

    for alpha in alpha_list:

        w, b, cost_history = gradient_descent(x, y, w0, b0, alpha, num_iters)

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
            "status": status,
            "final": {"w": w, "b": b},
            "gap_to_normal": final_cost - normal_cost
        })


        print(f"alpha: {alpha}, cost: {cost_history[-1]}")

    payload = {
        "meta": meta,
        "results": results
    }
    with open("./experiments/alpha_finder/results/results.json", "w") as f:
        json.dump(payload, f, indent=4)

def clean_data(data_file: str):
    hours = []
    scores = []
    row_skipped = 0
    with open(data_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                if not row["Scores"] or not row["Hours"]:
                    raise ValueError
                hours.append(float(row["Hours"]))
                scores.append(float(row["Scores"]))
            except ValueError:
                row_skipped += 1
                continue
    return hours, scores, row_skipped


if __name__ == "__main__":
    main()
