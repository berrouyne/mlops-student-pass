import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# Tiny dataset example (replace later with real data)
X = np.array([
    [2, 0.50, 8],
    [6, 0.90, 14],
    [4, 0.70, 10],
    [8, 0.95, 16],
    [1, 0.40, 6],
    [7, 0.80, 13],
    [3, 0.60, 9],
])

y = np.array([0, 1, 0, 1, 0, 1, 0])  # 0=FAIL, 1=PASS

model = LogisticRegression()
model.fit(X, y)

Path("model").mkdir(parents=True, exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Trained model saved to model/model.pkl")
