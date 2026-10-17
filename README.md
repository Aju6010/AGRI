# Crop Classification and Wheat Yield Prediction

This project implements the base paper's two-part machine learning workflow:

- Crop recommendation from soil and weather features using robust classifiers.
- Wheat yield prediction for Pakistan using MICE-style imputation and regression models.

The app works immediately with deterministic sample data. If you have the real datasets, place them in `data/raw` and retrain.

## Project Structure

```text
agri_ml/
  config.py        Feature names, paths, crop labels
  data.py          Dataset loading, schema normalization, sample data generation
  models.py        Classifier/regressor training, metrics, feature importance
  train.py         CLI training entry point
web_app.py         Flask product web app
web/               HTML, CSS, and JavaScript product interface
app.py             Streamlit application
requirements.txt   Python dependencies
tests/             Smoke tests
```

## Datasets

### Crop recommendation

Expected CSV columns:

```text
N, P, K, temperature, humidity, ph, rainfall, label
```

Save as:

```text
data/raw/crop_recommendation.csv
```

### Wheat yield prediction

The loader supports the common Kaggle crop-yield schema and filters Pakistan wheat rows when `Area` and `Item` are available.

Expected useful columns:

```text
Area, Item, Year, hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
```

Save as:

```text
data/raw/crop_yield.csv
```

If either file is missing, the project uses generated sample data so the workflow stays runnable.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train models:

```powershell
python -m agri_ml.train
```

Launch the product web app:

```powershell
python web_app.py
```

Open:

```text
http://localhost:5000
```

Optional legacy Streamlit app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
pytest
```

## Implemented Paper Methods

- Classification models: Extra Trees, Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Gaussian Naive Bayes, Support Vector Machine.
- Regression models: Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Regressor, XGBoost when available with a Gradient Boosting fallback.
- Preprocessing: schema cleaning, feature scaling where needed, train/test split, 5-fold cross-validation, MICE-style iterative imputation for yield data.
- Metrics: accuracy, precision, recall, F1, confusion matrix, R2, MAE, MSE, RMSE, standard deviation of absolute errors.
- Explainability: model feature importance where supported, with permutation-importance fallback.
