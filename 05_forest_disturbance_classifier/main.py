import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(75)
n = 1600
rainfall = rng.uniform(0, 300, n)
temperature = rng.uniform(18, 42, n)
vegetation = rng.uniform(0, 1, n)
road_activity = rng.uniform(0, 1, n)
human_pressure = rng.uniform(0, 1, n)
risk = (0.25 * temperature / 42 + 0.25 * (1 - vegetation) + 0.25 * road_activity + 0.25 * human_pressure - rainfall / 1200)
labels = np.where(risk > 0.62, 2, np.where(risk > 0.43, 1, 0))
X = np.column_stack([rainfall, temperature, vegetation, road_activity, human_pressure])
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=75, stratify=labels)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=300, random_state=75, early_stopping=True))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, zero_division=0))
os.makedirs("output", exist_ok=True)
pd.DataFrame(X_test, columns=["rainfall", "temperature", "vegetation", "road_activity", "human_pressure"]).assign(actual_risk=y_test, predicted_risk=pred).to_csv("output/forest_risk_predictions.csv", index=False)
print("Note: generated data is simulated, not satellite observations.")
