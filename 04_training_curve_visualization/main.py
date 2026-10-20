import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1400, n_features=16, n_informative=9, random_state=44)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=250, random_state=44, early_stopping=True))
model.fit(X_train, y_train)
network = model.named_steps["mlpclassifier"]
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(network.loss_curve_)
plt.xlabel("Training iteration")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss")
plt.tight_layout()
plt.savefig("output/training_loss.png", dpi=150)
plt.show()
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
