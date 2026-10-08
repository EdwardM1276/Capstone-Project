# Models

The training workflow saves each fitted model here and exports the highest
test-set PR-AUC model as `best_model.joblib` for the Streamlit application.
Model binaries are generated locally because their size depends on the dataset
and selected strategy. Run the training workflow before launching the app.
