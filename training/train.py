from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from pathlib import Path

X = np.array([
    [2, 0.50, 10],
    [6, 0.90, 14],
    [4, 0.70, 12],
    [8, 0.95, 16],
    [1, 0.40, 8],
    [7, 0.85, 15],
    [3, 0.60, 11]
])

y = np.array([0, 1, 0, 1, 0, 1, 0])

model = LogisticRegression(
    solver="liblinear",
    max_iter=200
)

model.fit(X, y)

Path("model").mkdir(exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
