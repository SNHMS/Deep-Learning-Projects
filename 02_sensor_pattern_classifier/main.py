import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_classification(
    n_samples=1800, n_features=12, n_informative=8, n_redundant=2,
    n_classes=3, n_clusters_per_class=2, class_sep=1.2, random_state=22
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22, stratify=y)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=300, random_state=22, early_stopping=True))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
os.makedirs("output", exist_ok=True)
pd.DataFrame({"actual_class": y_test, "predicted_class": pred}).to_csv("output/sensor_pattern_predictions.csv", index=False)
