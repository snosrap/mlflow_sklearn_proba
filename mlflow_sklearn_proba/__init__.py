import os
import platform
import cloudpickle
import mlflow

class PythonModelProba(mlflow.pyfunc.PythonModel):
  def __init__(self):
    self.sk_model = None
  def load_context(self, context):
    with open(context.artifacts["pickle_path"], "rb") as f:
      self.sk_model = cloudpickle.load(f)
  def predict(self, context, model_input):
    return self.sk_model.predict_proba(model_input)

def save_model(sk_model, path, **kwargs):
  # Save the model using sklearn flavor
  mlflow.sklearn.save_model(sk_model, path, **kwargs)

  # Load the model back up from disk
  saved_mlflow_model = mlflow.models.Model.load(os.path.join(path, mlflow.models.model.MLMODEL_FILE_NAME))

  # Save a PyFunc pickle (based on `PythonModelProba` above)
  with open(os.path.join(path, "python_model.pkl"), "wb") as f:
    cloudpickle.dump(PythonModelProba(), f)

  # Overwrite default SKLearn PyFunc flavor with our own
  saved_mlflow_model.add_flavor(
    "python_function",
    artifacts = {
      "pickle_path": {
        "path": "model.pkl",
        "uri": "model.pkl",
      },
    },
    cloudpickle_version = cloudpickle.__version__,
    env = "conda.yaml",
    loader_module = "mlflow.pyfunc.model",
    python_model = "python_model.pkl",
    python_version = platform.python_version()
  )

  # Save the model to disk again
  saved_mlflow_model.save(os.path.join(path, mlflow.models.model.MLMODEL_FILE_NAME))
  
def log_model(sk_model, artifact_path, **kwargs):
  kwargs["sk_model"] = sk_model
  return mlflow.models.Model.log(artifact_path, flavor=mlflow_sklearn_proba, **kwargs)
