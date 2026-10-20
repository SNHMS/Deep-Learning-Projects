import os
import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=3000, n_features=18, n_informative=10, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
results = []
for name, layers in [("Small network", (16,)), ("Deep network", (32, 16, 8))]:
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=250, random_state=12, early_stopping=True)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    results.append({"model": name, "hidden_layers": str(layers), "seconds": seconds, "accuracy": accuracy_score(y_test, model.predict(X_test))})
result = pd.DataFrame(results)
os.makedirs("output", exist_ok=True)
result.to_csv("output/network_benchmark.csv", index=False)
print(result.round(3).to_string(index=False))
