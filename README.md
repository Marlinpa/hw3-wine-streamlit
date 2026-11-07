# Wine Quality — HW3

Streamlit demo for a wine quality classifier. A saved StandardScaler + LogisticRegression
pipeline estimates whether a wine's expert rating is at least 7/10.

## Run locally

Use Python 3.13, then run:

```sh
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```

## Streamlit Community Cloud

- Repository: `Marlinpa/hw3-wine-streamlit`
- Branch: `main`
- Main file: `wine_app/app.py`
- Python: `3.13`

All model artifacts and the held-out test set are included. The threshold was selected
on a validation subset of the training data. See wine_app/README.md for details.
