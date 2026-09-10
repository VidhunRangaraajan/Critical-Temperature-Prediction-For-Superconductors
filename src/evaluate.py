# evaluate.py
# Description: 
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Critical-Temperature-Prediction-For-Superconductors
# Requirements: 
# Usage: 
# Depends On: 
# Input Files: 
# Output Files: 
# Notes: 
# To Do: 

# Importing necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Loading the datasets.
df = pd.read_csv("data/cleaned_critical_temp.csv")
x_test = pd.read_csv("data/x_test_s.csv")
y_test = pd.read_csv("data/y_test.csv")
y_mean = pd.read_csv("results/y_mean.csv")

# Plotting the correlation heatmap of the cleaned dataset.
df_corr = df.corr()
plt.figure(figsize=(60, 50))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig("results/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# Saving the model's file path in a list.
files = [
    "results/ExtraTreesRegressor_params.joblib",
    "results/RandomForestRegressor_params.joblib",
    "results/KNeighborsRegressor.joblib",
    "results/DecisionTreeRegressor.joblib",
    "results/GradientBoostingRegressor.joblib",
]

# Loading the models and saving it in a dictionary.
models = {}
for file in files:
    model_name = file.replace(".joblib", "").replace("results/", "").replace("_params", "")
    models[model_name] = joblib.load(file)
    
    # Rebuilding the model from the trained parameters for the large models.
    if model_name in ["ExtraTreesRegressor","RandomForestRegressor"]:
        
        if model_name == "RandomForestRegressor":
            _model = RandomForestRegressor(**models[model_name]["hyperparams"])
            
        elif model_name == "ExtraTreesRegressor":
            _model = ExtraTreesRegressor(**models[model_name]["hyperparams"])

        # Attach the trained estimators
        _model.estimators_ = models[model_name]["estimators"]
        
        # Adding the number of outputs and number of features in the rebuilt model.
        _model.n_outputs_ = 1
        _model.n_features_in_ = 81

        # Save the model like the other models in the dictionary.
        models[model_name] = _model
    
# Function to return the predictions of the loaded model which is given in parameters.
def predict(model):
    y_pred = model.predict(x_test)
    return y_pred.ravel()

# Evaluating the models.
metrics = []
for name, model in models.items():
    
    # Printing the name of the model being evaluated and finding the predicted value for the test dataset.
    print(f"Evaluating {name}...")
    y_pred = model.predict(x_test)

    # Taking evaluation metrics.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics.append([name, rmse, mae, r2])

# Printing the name of the model being evaluated.
print("Evaluating mean of predictions...")

# Taking evaluation metrics.
rmse = np.sqrt(mean_squared_error(y_test, y_mean))
mae = mean_absolute_error(y_test, y_mean)
r2 = r2_score(y_test, y_mean)
metrics.append(["Mean of Predictions", rmse, mae, r2])

# Saving the metrics to a CSV file.
df = pd.DataFrame(metrics, columns=["Model", "RMSE", "MAE", "R²"])
df.to_csv("results/metrics.csv", index=False)