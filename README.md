# Stellar Classification Model

## Description

This project develops a machine learning model for classifying celestial objects (stars, galaxies, and quasars) based on astronomical data. The Jupyter Notebook `stellar_classification(1)(1).ipynb` walks through the entire process from data loading and exploration to model deployment.

## Data

The dataset used is `stellar_classification.csv`. It contains various astronomical measurements for a large number of celestial objects.

## Preprocessing & Exploratory Data Analysis

The notebook performs the following data preprocessing steps:
* Initial inspection of data shape, information, and column names.
* Verification of no missing or duplicated data.
* Dropping irrelevant columns such as `obj_ID`, `run_ID`, `rerun_ID`, `field_ID`, `fiber_ID`, `spec_obj_ID`, `MJD`, and `cam_col`.
* Analysis of data distributions and correlations using Pearson and Spearman correlation heatmaps and histograms.
* Identification and removal of an extreme outlier where `u`, `g`, and `z` values were -9999.0.
* Transformation of the `redshift` column using `np.log1p` to handle its heavy-tailed distribution.

## Feature Engineering

Astronomical colors are created by subtracting the magnitude of an object in one filter from its magnitude in another. The following new features are engineered:
* `u_minus_g` (Ultraviolet - Green)
* `g_minus_r` (Green - Red)
* `r_minus_i` (Red - Near Infrared)
* `u_minus_r` (Ultraviolet - Red)
* `g_minus_i` (Green - Near Infrared)

## Modeling

The project explores several different models:
**SGD Classifier (Stochastic Gradient Descent)**: Used with OneVsRestClassifier for multi-class classification.
**Logistic Regression**: Implemented with elasticnet penalty and optimized hyperparameters.
**Naive Bayes (GaussianNB)**: Used as another baseline classification model.
**Random Forest Classifier**: Tuned and evaluated, showing strong performance.
**XGBoost Classifier**: Also tuned and evaluated, demonstrating competitive results.

## Results

Both XGBoost and Random Forest Classifiers performed exceptionally well and showed consistent and great accuracy. The **Random Forest Classifier** marginally outperformed XGBoost in overall test accuracy and per-class F1-scores thus making it the slightly more optimal classifier for this dataset based on these metrics.

## Deployment

The best-performing Random Forest pipeline is saved using `joblib` as `my_classification_model.joblib` for future deployment.

## Usage

To use this project:
1.  Clone the repository.
2.  Install the necessary Python libraries (e.g., scikit-learn, pandas, numpy, seaborn, matplotlib, joblib).
3.  Ensure `stellar_classification.csv` is in the project directory.
4.  Run the `stellar_classification(1)(1).ipynb` notebook in a Jupyter environment.
