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

# Importing the required libraries.
import pandas as pd
import joblib
import statistics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

# Loading the test data from the CSV files.
x_test = pd.read_csv("data/x_test_s.csv")
y_test = pd.read_csv("data/y_test.csv")

# Saving the model's file path in a list.
files = [
    "results/ExtraTreesRegressor_params.joblib",
    "results/RandomForestRegressor_params.joblib",
    "results/KNeighborsRegressor.joblib",
    "results/DecisionTreeRegressor.joblib",
]

# List of feature columns used in the model.
feature_cols = ["number_of_elements","mean_atomic_mass","wtd_mean_atomic_mass","gmean_atomic_mass","wtd_gmean_atomic_mass","entropy_atomic_mass","wtd_entropy_atomic_mass","range_atomic_mass","wtd_range_atomic_mass","std_atomic_mass","wtd_std_atomic_mass","mean_fie","wtd_mean_fie","gmean_fie","wtd_gmean_fie","entropy_fie","wtd_entropy_fie","range_fie","wtd_range_fie","std_fie","wtd_std_fie","mean_atomic_radius","wtd_mean_atomic_radius","gmean_atomic_radius","wtd_gmean_atomic_radius","entropy_atomic_radius","wtd_entropy_atomic_radius","range_atomic_radius","wtd_range_atomic_radius","std_atomic_radius","wtd_std_atomic_radius","mean_Density","wtd_mean_Density","gmean_Density","wtd_gmean_Density","entropy_Density","wtd_entropy_Density","range_Density","wtd_range_Density","std_Density","wtd_std_Density","mean_ElectronAffinity","wtd_mean_ElectronAffinity","gmean_ElectronAffinity","wtd_gmean_ElectronAffinity","entropy_ElectronAffinity","wtd_entropy_ElectronAffinity","range_ElectronAffinity","wtd_range_ElectronAffinity","std_ElectronAffinity","wtd_std_ElectronAffinity","mean_FusionHeat","wtd_mean_FusionHeat","gmean_FusionHeat","wtd_gmean_FusionHeat","entropy_FusionHeat","wtd_entropy_FusionHeat","range_FusionHeat","wtd_range_FusionHeat","std_FusionHeat","wtd_std_FusionHeat","mean_ThermalConductivity","wtd_mean_ThermalConductivity","gmean_ThermalConductivity","wtd_gmean_ThermalConductivity","entropy_ThermalConductivity","wtd_entropy_ThermalConductivity","range_ThermalConductivity","wtd_range_ThermalConductivity","std_ThermalConductivity","wtd_std_ThermalConductivity","mean_Valence","wtd_mean_Valence","gmean_Valence","wtd_gmean_Valence","entropy_Valence","wtd_entropy_Valence","range_Valence","wtd_range_Valence","std_Valence","wtd_std_Valence"]

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

# Saving each model's prediction in a variable.
ETR = predict(models["ExtraTreesRegressor"])
RFR = predict(models["RandomForestRegressor"])
KNR = predict(models["KNeighborsRegressor"])
DTR = predict(models["DecisionTreeRegressor"])

# Finding the mode of the predicted values from all the models.
y_mean = []
for i in range(len(ETR)):
    j=statistics.mean([ETR[i],RFR[i],KNR[i],DTR[i]])
    y_mean.append(j)

# Saving the mode of the predicted values in a CSV file.
df = pd.DataFrame(y_mean, columns=["Prediction"])
df.to_csv("results/y_mean.csv", index=False)