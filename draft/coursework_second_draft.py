#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:45 2024

@author: charlieunsworth
"""

# import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
import random


# import datasets

tasks = pd.read_csv('tasks.csv')
suppliers = pd.read_csv('suppliers.csv')
cost = pd.read_csv('cost.csv')

# Check df 1. tasks
tasks.head(1)

tasks.info()

# Check df 2. suppliers
suppliers.head(1)

# Check df 3. cost
cost.head(1)

cost.info()

cost.groupby('Supplier ID').describe()

# Missing values
print('Missing values in suppliers dataset:', suppliers.isnull().sum().sum())
print('Missing values in cost dataset:', cost.isnull().sum().sum())
print('Missing values in tasks dataset:', tasks.isnull().sum().sum())

# Tasks (unrelated to cost)

print("No of Task ID in Tasks:",tasks['Task ID'].nunique())
print("No of Task ID in cost:",cost['Task ID'].nunique())

cost_match = cost['Task ID'].unique()

tasks=tasks.loc[tasks['Task ID'].isin(cost_match)]

tasks['Task ID'].nunique()

print("'tasks' dataset info:\n{} tasks\n{} task features".format(len(tasks),len(tasks.columns[1:])))
#suppliers
print("\n'suppliers' dataset info:\n{} suppliers\n{} supplier features".format(len(suppliers),len(suppliers.columns[1:])))
#costs
print("\n'cost' dataset info:\n{} tasks\n{} suppliers\n{} cost values".format(len(cost.loc[:,'Task ID'].unique()), \
    len(cost.loc[:,'Supplier ID'].unique()), len(cost.loc[:, 'Cost'])))

# Fixing the structure of the suppliers df
suppliers = suppliers.transpose()
suppliers.columns = suppliers.iloc[0]
suppliers = suppliers.drop('Features', axis='index')
suppliers = suppliers.reset_index()

# Fixing the non-numeric columns in Tasks
tasks = tasks.set_index('Task ID')
for col in tasks.columns:
    if tasks[col].dtype == 'object':  # Check if column is of object type (often strings)
        try:
            # Attempt to convert to numeric, handling errors
            tasks[col] = pd.to_numeric(tasks[col].str.rstrip('%'), errors='coerce') /100
        except AttributeError:
            pass # Skip if not a string or does not contain '%'
tasks = tasks.reset_index()
    
# Feature Importance

from sklearn.ensemble import RandomForestRegressor

# Combine datasets for quick analysis (without costs for now)
combined = tasks.merge(suppliers, how='cross')

# Placeholder target variable (use real costs in actual analysis)

combined = combined.merge(cost, left_on=['Task ID', 'index'], right_on=['Task ID','Supplier ID'])
combined_features = combined.drop(['Task ID','index','Supplier ID', 'Cost'],  axis='columns')
combined_y = combined['Cost']


# Fit a Random Forest for feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(combined_features, combined_y)

# Get feature importances
importances = rf.feature_importances_
feature_names = combined_features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

importance_df.head(10)

importance_df.info()

plt.boxplot(importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Box Plot of Feature Importance')
plt.show()

importance_df['Importance'].describe()    
    
    
# Remove under 25% of Importance
importance_threshold = importance_df['Importance'].quantile(q=0.25)
importance_df = importance_df[importance_df['Importance'] >= importance_threshold]
    
# Check the feature list
feature_list = importance_df['Feature'].unique()
feature_list

# Feature Seperate
features_T = [feature for feature in feature_list if feature.startswith('T')]
features_S = [feature for feature in feature_list if feature.startswith('S')]

print('Tasks Features:', features_T)
print('Suppliers Features:', features_S)    

#Reinstate indicies
tasks = tasks.set_index('Task ID')
suppliers = suppliers.set_index('index')
suppliers.index.rename('Supplier ID', inplace=True)
    
#Select the features ahead of scaling
suppliers_selected = suppliers[features_S]
tasks_selected = tasks[features_T]


# Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
tasks_selected[:] = scaler.fit_transform(tasks_selected)
suppliers_selected[:] = scaler.fit_transform(suppliers_selected)
    
# Top-performing
avg_cost = cost.groupby('Supplier ID')['Cost'].mean()
avg_cost.sort_values()
avg_cost.head(n=5)

plt.boxplot(avg_cost)
plt.xlabel('Mean Cost')
plt.title('Box Plot of Mean Cost by Supplier')
plt.show()

# Worst-performing ((modifiable) Standard : 10%)
cost_threshold = avg_cost.quantile(q=0.95)

chosen_suppliers = avg_cost[avg_cost.values < cost_threshold]

suppliers_scaled_trimmed = suppliers_selected[suppliers.index.isin(chosen_suppliers.index)]

# Using this threshold, 4 suppliers were trimmed from the dataset
print('Suppliers removed:', len(suppliers)-len(suppliers_scaled_trimmed))

"""# 2. EDA
"""


# 3. ML model fitting and scoring
# 3.1 Removing our trimmed suppliers from the costs dataset and combining the three datasets into a single df

cost_preprocessed = cost[cost['Supplier ID'].isin(chosen_suppliers.index)]
tasks_preprocessed = tasks_selected.reset_index()
suppliers_preprocessed = suppliers_scaled_trimmed.reset_index()

merged = tasks_preprocessed.merge(suppliers_preprocessed, how='cross')
dataset = merged.merge(cost_preprocessed, 
                       left_on=['Task ID','Supplier ID'],
                       right_on=['Task ID','Supplier ID'])

# 3.2 Splitting into test and train datasets, manually to preserve the groups
X = dataset.drop(['Task ID','Supplier ID','Cost'], axis='columns')
y = dataset['Cost']
Groups = dataset['Task ID']

all_tasks = Groups.unique()
random.seed(42)
TestGroup = np.random.choice(all_tasks, size=20, replace=False)

test_loc = dataset['Task ID'].isin(list(TestGroup))

X_test = X[test_loc]
y_test = y[test_loc]
X_train = X[~test_loc]
y_train = y[~test_loc]

# 3.3 Fitting a ridge regression model
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge.score(X_test,y_test)
y_pred = ridge.predict(X_test)

# 3.4 Grouping by tasks and selecting a supplier for each test task
# using the error formula provided calculate difference between true supplier and predicted supplier cost

# Getting Task IDs for our predictions
ids = dataset['Task ID'][test_loc].reset_index(drop=True)
y_pred = pd.Series(y_pred, name="Predicted Cost")

y_pred_ids = pd.merge(ids, y_pred, right_index = True, left_index = True)
pred_best_suppliers = y_pred_ids.groupby('Task ID')['Predicted Cost'].min()

# Getting the true cheapest suppliers for our test set
true_best_suppliers = cost.groupby('Task ID')['Cost'].min()

test_best_suppliers = true_best_suppliers[true_best_suppliers.index.isin(TestGroup)]

# Defining functions to calculate errors and rmse
def error_calc(true_min_costs, predicted_min_cost):
   error = true_min_costs - predicted_min_cost
   return error

def rmse_calc(error_array):
    squared_errors = error_array*error_array
    rss = np.sum(squared_errors)
    value = np.sqrt(rss/len(error_array))
    return value

ridge_error = error_calc(test_best_suppliers, pred_best_suppliers)
ridge_rmse = rmse_calc(ridge_error)

# Results for an initial ridge model
print(ridge_error)
print(ridge_rmse)

# 4 Performing cross validation
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer

train_tasks = dataset['Task ID'][~test_loc]

ridge_cv = Ridge(alpha=0.5)
logo = LeaveOneGroupOut()

# Defining our scoring function:
# Since the test set for each fold of the cross validation only contains one group
# we can just find the minimums of each array, then compute rmse from that

def error_scoring(all_trues, all_preds):
    error = all_trues.min() - all_preds.min()
    return error
error_scorer = make_scorer(error_scoring)

scores = cross_val_score(ridge_cv, X_train, y_train, cv=logo, groups=train_tasks, scoring=error_scorer)
cv_rmse = rmse_calc(scores)
