from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib, os

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
out_path = os.path.join(os.path.dirname(__file__), "iris_lr.joblib")
joblib.dump(clf, out_path)
print("Saved", out_path)
