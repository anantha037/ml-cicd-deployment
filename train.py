from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import sys

version = sys.argv[1] if len(sys.argv) > 1 else "v1"

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(model,f"model_{version}.joblib")

print(f"Model saved: model_{version}.joblib")