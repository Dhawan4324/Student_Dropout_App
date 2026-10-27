
# Student Academic Outcome Predictor

## Files

- app.py: Streamlit application
- model.joblib: Frozen XGBoost preprocessing pipeline
- encoder.joblib: Target label encoder
- config.json: Feature schema and model settings
- requirements.txt: Python dependencies

## Local execution

pip install -r requirements.txt

streamlit run app.py

## Streamlit Community Cloud

1. Create a GitHub repository for this application.
2. Upload the contents of this folder to its root.
3. Open https://share.streamlit.io/
4. Create a new application using the repository.
5. Set the main Python file to app.py.
6. Deploy the application.

## Important

The model is a research prototype.

Use only trusted model.joblib and encoder.joblib files.

The application uses the frozen research model and
does not update model parameters after prediction.
