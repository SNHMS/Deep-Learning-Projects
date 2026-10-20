import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(86)
n = 900
temperature = rng.uniform(18, 40, n)
occupancy = rng.uniform(0, 1, n)
hour = rng.uniform(0, 24, n)
devices = rng.integers(1, 15, n)
energy = 0.15 * temperature + 2.4 * occupancy + 0.06 * devices + 0.9 * np.sin(hour * np.pi / 12) + rng.normal(0, 0.35, n)
X = pd.DataFrame({"temperature": temperature, "occupancy": occupancy, "hour": hour, "devices": devices})
y = energy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)
model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=86, early_stopping=True))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("R2:", round(r2_score(y_test, pred), 3))
os.makedirs("output", exist_ok=True)
pd.DataFrame({"actual_energy": y_test, "predicted_energy": pred}).to_csv("output/energy_regression.csv", index=False)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.7)
plt.xlabel("Actual energy")
plt.ylabel("Predicted energy")
plt.title("Actual vs Predicted Energy")
plt.tight_layout()
plt.savefig("output/energy_prediction_plot.png", dpi=150)
plt.show()
