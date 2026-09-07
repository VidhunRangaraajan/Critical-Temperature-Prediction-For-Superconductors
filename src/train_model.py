# train_model.py
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

# Importing required libraries.
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump

# Loading the training datasets.
x_train_s = pd.read_csv("data/x_train_s.csv")
y_train = pd.read_csv("data/y_train.csv")

# Creating a dictionary of models.
models = {
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=12, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

# Training each model and saving the trained model.
for name, model in models.items():
    
    # Training the model.
    print(f"Training {name}...")# Displays the name of the model being trained in the terminal.
    model.fit(x_train_s, y_train)
    
    # Saving the trained model.
    dump(model, f"results/{name}.joblib")