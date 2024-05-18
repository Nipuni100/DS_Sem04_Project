import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

df_emp = pd.read_csv('./employees.csv')

# Since the Employee_No is a primary key. We set it as the index.
df_emp.set_index('Employee_No', inplace=True)

# Employee_code and Name are not important for a machine learning model
df_emp.drop(columns=['Employee_Code'], inplace=True)
df_emp.drop(columns=['Name', 'Title'], inplace=True)

# Encoding gender
df_emp["Gender"] = df_emp["Gender"].map({"Male": 1, "Female": 0})

# Religion already has an id column
df_emp.drop(columns=['Religion'], inplace=True)

# Imputing marital status using the mode
most_frequent_marital_status = df_emp['Marital_Status'].mode()[0]
df_emp['Marital_Status'] = df_emp['Marital_Status'].fillna(
    most_frequent_marital_status)

# Imputing year of birth using a knn tree
df_emp['Year_of_Birth'] = pd.to_numeric(
    df_emp['Year_of_Birth'], errors='coerce')
imputer = KNNImputer(n_neighbors=1)
df_emp['Year_of_Birth'] = imputer.fit_transform(df_emp[['Year_of_Birth']])

# Encoding marital status
df_emp["Marital_Status"] = df_emp["Marital_Status"].map(
    {"Married": 1, "Single": 0})

# Designation already has an id column
df_emp.drop(columns=['Designation'], inplace=True)

# Decoding joined date and inactive date
df_emp['Date_Joined'] = pd.to_datetime(
    df_emp['Date_Joined'], format='%m/%d/%Y')
df_emp['Joined_Date'] = df_emp['Date_Joined'].apply(lambda date: date.day)
df_emp['Joined_Month'] = df_emp['Date_Joined'].apply(lambda date: date.month)
df_emp['Joined_Year'] = df_emp['Date_Joined'].apply(lambda date: date.year)

df_emp.drop(columns=['Date_Joined'], inplace=True)
df_emp.drop(columns=['Date_Resigned'], inplace=True)

df_emp['Inactive_Date'].replace(['\\N', '0000-00-00'], pd.NA, inplace=True)

df_emp['Inactive_Date'] = pd.to_datetime(
    df_emp['Inactive_Date'], format='%m/%d/%Y', errors='coerce')

df_emp['Inactive_Day'] = df_emp['Inactive_Date'].dt.day
df_emp['Inactive_Month'] = df_emp['Inactive_Date'].dt.month
df_emp['Inactive_Year'] = df_emp['Inactive_Date'].dt.year

df_emp.drop(columns=['Inactive_Date'], inplace=True)

# Dropping Reporting_emp_2 as all values are null
df_emp.drop(columns=["Reporting_emp_2"], inplace=True)

# Encoding employment category
status_encoded_column = pd.get_dummies(
    df_emp['Employment_Category'], prefix='Employment_Category')
df_emp = df_emp.drop('Employment_Category', axis=1)
df_emp = df_emp.join(status_encoded_column)

# Encoding status
mapping = {'Active': 1, 'Inactive': 0}
df_emp["Status"] = df_emp["Status"].map(mapping)
df_emp.rename(columns={"Status": "Status"}, inplace=True)
mapping = {'Permanant': 1, 'Contarct Basis': 0}

# Encoding employment type
df_emp["Employment_Type"] = df_emp["Employment_Type"].map(mapping)
df_emp.rename(columns={"Employment_Type": "Employment_Type"}, inplace=True)
df_emp = df_emp.replace(True, 1)
df_emp = df_emp.replace(False, 0)

# Dropping Reporting_emp_1 as it has majority of null values
df_emp.drop(columns=['Reporting_emp_1'], inplace=True)

# Converting year of birth to int
df_emp['Year_of_Birth'] = df_emp['Year_of_Birth'].astype(int)

# Saving the preprocessed dataset
df_emp.to_csv('employee_preprocess_group_03.csv', index=False)
