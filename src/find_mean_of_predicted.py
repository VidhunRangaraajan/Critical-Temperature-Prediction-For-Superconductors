# find_mean_of_predicted.py
# Description: This program finds the mean of the predicted values from multiple machine learning models and saves it in a CSV file.
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Critical-Temperature-Prediction-For-Superconductors
# Requirements: pandas, Scikit-learn, Joblib, statistics.
# Usage: To find the mean of the predictions.
# Depends On: -
# Input Files: data/x_test_s.csv, results/ExtraTreesRegressor_params.joblib, results/RandomForestRegressor_params.joblib, results/KNeighborsRegressor.joblib, results/DecisionTreeRegressor.joblib
# Output Files: results/y_mean.csv
# Notes: Run the preprocessing.py, followed by train_model.py first.
# To Do: -

# Importing the required libraries.
import pandas as pd
import joblib
import statistics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

# Loading the test data from the CSV files.
x_test = pd.read_csv("data/x_test_s.csv")

# Saving the model's file path in a list.
files = [
    "results/ExtraTreesRegressor_params.joblib",
    "results/RandomForestRegressor_params.joblib",
    "results/KNeighborsRegressor.joblib",
    "results/DecisionTreeRegressor.joblib",
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

        # Attaching the trained estimators.
        _model.estimators_ = models[model_name]["estimators"]
        
        # Adding the number of outputs and number of features in the rebuilt model.
        _model.n_outputs_ = 1
        _model.n_features_in_ = 81

        # Saving the model like the other models in the dictionary.
        models[model_name] = _model
    
# Function to return the predictions of the loaded model which is given in parameter.
def predict(model):
    y_pred = model.predict(x_test)
    return y_pred.ravel()

# Saving each model's prediction in a variable.
ETR = predict(models["ExtraTreesRegressor"])
RFR = predict(models["RandomForestRegressor"])
KNR = predict(models["KNeighborsRegressor"])
DTR = predict(models["DecisionTreeRegressor"])

# Finding the mean of the predicted values from all the models.
y_mean = []
for i in range(len(ETR)):
    j=statistics.mean([ETR[i],RFR[i],KNR[i],DTR[i]])
    y_mean.append(j)

# Saving the mean of the predicted values in a CSV file.
df = pd.DataFrame(y_mean, columns=["Prediction"])
df.to_csv("results/y_mean.csv", index=False)