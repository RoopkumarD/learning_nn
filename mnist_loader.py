import csv
import pickle

import numpy as np

filename = "mnist_train"

o = {"x": np.array([]), "y": np.array([])}


with open(f"{filename}.csv", "r") as f:
    data = csv.DictReader(f)
    y_values = []
    x_values = []
    for row in data:
        y_values.append(int(row["label"]))
        temp = []
        for v in row:
            if v != "label":
                temp.append(int(row[v]))
        x_values.append(temp)

    o["y"] = np.array(y_values)
    o["x"] = np.array(x_values)


with open(f"{filename}.pkl", "wb") as f:
    pickle.dump(o, f)
