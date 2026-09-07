# preprocessing.py
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Creating a dataframe from the csv file.
df = pd.read_csv("data/critical_temp.csv")

# Dropping duplicates and resetting the index.
df = df.drop_duplicates().reset_index(drop=True)

# Finding the feature columns by excluding the target column "critical_temp".
feature_cols = [i for i in df.columns if i != "critical_temp"]

# Creating the feature matrix (x) and target vector (y).
x = df[feature_cols].values
y = df["critical_temp"].values

# Splitting the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardizing columns.
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

# Saving the preprocessed data to csv files.
dump(scaler, "data/scaler.joblib")# Saving the scaler object for future use in model deployment.
df.to_csv('data/cleaned_critical_temp.csv', index=False)# Saving the cleaned data to a csv file.
pd.DataFrame(x_train_s).to_csv('data/x_train_s.csv', index=False)
pd.DataFrame(x_test_s).to_csv('data/x_test_s.csv', index=False)
pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)