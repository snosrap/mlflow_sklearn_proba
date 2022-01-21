import mlflow
from mlflow_sklearn_proba import save_model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Create and save 
X, y = load_iris(return_X_y=True)
model = LogisticRegression(random_state=0, max_iter=10000).fit(X, y)
save_model(model, "model_proba")

# load pyfunc flavor
model_proba_py = mlflow.pyfunc.load_model("model_proba")
print(model_proba_py.predict(X)) # Note: probabilities for each category instead of single category prediction

# load sklearn flavor
model_proba_sk = mlflow.sklearn.load_model("model_proba")
print(model_proba_sk.predict(X)) # pyfunc's old behavior
print(model_proba_sk.predict_proba(X)) # pyfunc's new behavior with mlflow_sklearn_proba
