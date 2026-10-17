# KV-Social-Media-Mining-and-Analysis
- All requirements are defined in the `pyproject.toml` and our exact package versions can be found in the `uv.lock` file.
- Models and training runs are configured using their respective *.jsons.
- The classifiers are trained using `uv run baseline.py <json-path>`.
- Outputs can be found in the outputs folder including detailed hyperparam configurations of the best model and all the shap outputs that were generated.
- Shap plots can be generated using `uv run explain_shap.py` the respective classifier here has to be set as a variable in the .py.
- Some parts of the .ipynb have been used to inspect the results.
