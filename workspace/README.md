# Sowing Success: Crop Recommendation via Logistic Regression

## Problem Statement
Help farmers pick the optimal crop for a field using soil nutrient profiles. The dataset contains nitrogen, phosphorous, potassium, and pH measurements paired with the known best-performing crop. The goal is to predict the crop given the soil chemistry and identify which nutrient is most predictive.

## Dataset Description
- `soil_measures.csv` includes the following columns:
  - `N`, `P`, `K`: soil nutrient ratios
  - `pH`: soil acidity
  - `crop`: categorical label representing the recommended crop for the field

## Methodology
1. Inspect the dataset for missing values and confirm numeric types for predictors.
2. Split the data (70/30) into training/testing sets with a fixed random seed for reproducibility.
3. Scale features using `StandardScaler` and fit a multinomial logistic regression.
4. Evaluate the model with balanced accuracy and record the nutrient that delivers the highest score in a feature selection loop.
5. Pair numeric summaries with plots (histograms, pairplots) to contextualize the predictive signal.

## Key Findings
- The notebook reports that potassium appears to be the most predictive nutrient, with balanced accuracy near 0.29 across the test set.
- Visualizations show nutrient distributions and class separations, helping non-technical stakeholders understand model behavior.

## Technologies Used
- Python 3.9+
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. `pip install -r requirements.txt` from the `workspace` folder.
2. Launch `notebook.ipynb` in Jupyter.
3. Confirm that `soil_measures.csv` sits beside the notebook.

## Future Improvements
- Experiment with tree-based models (RandomForest, XGBoost) to compare against logistic regression.
- Use k-fold cross-validation to stabilize the balanced accuracy score.
- Add SHAP or permutation importance plots to further explain nutrient impact.
