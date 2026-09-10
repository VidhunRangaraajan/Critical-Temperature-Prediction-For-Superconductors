# predict.py
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
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

# Saving the model's file path in a list.
files = [
    "results/ExtraTreesRegressor_params.joblib",
    "results/RandomForestRegressor_params.joblib",
    "results/KNeighborsRegressor.joblib",
    "results/DecisionTreeRegressor.joblib",
    "results/GradientBoostingRegressor.joblib",
]

# Path of scaler.
SCALER_FILE = "data/scaler.joblib"

# Module-level caches so models/scaler are loaded from disk only once, even if predict_mode() is called many times from another file.
models = None
scaler = None

# When the function is called for the first time, it loads and caches the 5 trained models and returns them as a dictionary.
def _load_models():
    global models
    if models is None:
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
    return models

# When the function is called for the first time, it loads and caches the fitted StandardScaler and returns it.
def _load_scaler():
    global scaler
    if scaler is None:
        scaler = joblib.load(SCALER_FILE)
    return scaler

# Preprocesses the input data (either a dictionary or a DataFrame) to match the format expected by the trained models and returns the preprocessed data.
def preprocess(raw_data):
    
    # Ensure scaler is loaded
    scaler = _load_scaler()
    
    # Creating a DataFrame from the input data.
    df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else raw_data.copy()
    
    # Scaling the DataFrame using the fitted StandardScaler.
    df = scaler.transform(df)
    # Returning the preprocessed DataFrame ready for prediction.
    return df

# Predicts the mean of predictions from all 5 trained models for the given input data(cleaned or runcleaned) and returns the prediction result along with the predictions of each model.
def predict_mean(input_data, already_preprocessed=False):
    
    # Preprocess the input data if it hasn't been preprocessed yet.
    df = input_data if already_preprocessed else preprocess(input_data)

    # Loading the trained models and making predictions for each model.
    models = _load_models()
    predictions = pd.DataFrame({name: model.predict(df).ravel() for name, model in models.items()})
    predictions_list = predictions.values.tolist()
    y_mean = predictions.mean(axis=1)[0].tolist()

    # Returning the prediction result based on the mean of predictions from all models and the predictions of each model.
    return [y_mean,] + predictions_list[0]
    
# Quick manual test using the n'th row row of the saved test set.
if __name__ == "__main__":
    n = int(input("Enter the row number of the test set to predict (0-100): "))
    x_test = pd.read_csv("data/x_test_s.csv")
    result = predict_mean(x_test.iloc[[n-2]], already_preprocessed=True)
    print(f"Prediction: {result}")