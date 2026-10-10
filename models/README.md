# Models

The training workflow saves each fitted model here and exports the model
selected by validation weighted error cost as `best_model.joblib` for the
Streamlit application. The final test set is reserved for reporting and does
not select the exported model. Model binaries are generated artifacts tied to
a specific data file and workflow version. Rerun training after changing the
data or methodology before using the application. Their size depends on the
dataset and selected strategy.
