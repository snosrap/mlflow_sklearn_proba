import setuptools

setuptools.setup(
  name = 'mlflow_sklearn_proba',
  version = "0.0.1",
  author = "Ford Parsons",
  description = "MLFlow SKLearn flavor that uses `predict_proba` instead of `predict` for pyfunc.predict while retaining the ability to load the sklearn flavor",
  packages = setuptools.find_packages(),
  python_requires = ">=3.7",
  install_requires = ["mlflow", "scikit-learn", "cloudpickle"],
)
