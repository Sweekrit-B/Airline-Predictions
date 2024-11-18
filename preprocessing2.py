#%% Import functions
#Import all necessary libraries into code

#Initial imports
import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib as plt
import seaborn as sns

#Ordinal encoding and simple imputation import
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

#Train test split import
from sklearn.model_selection import train_test_split

#Plotting and metrics imports
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report

#Random forest imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#XGBoost import
from xgboost import XGBClassifier

#%% Combining train and test
#Combines train and test CSVs to perform random splits + cross vals
airline_1 = pd.read_csv('train.csv')
airline_2 = pd.read_csv('test.csv')
combined_airline = pd.concat([airline_1, airline_2])

#%% Filling na + dropping unneeded columns
#Fills N/A values and drop 'Unnamed: 0' and 'id' columsn
combined_airline = combined_airline.fillna(0)
combined_airline = combined_airline.drop(columns=['Unnamed: 0', 'id'])
combined_airline

#%% Scaling flight distance 
#Use .apply(np.log) to scale flight distance to be natural log to decrease right skew of data
combined_airline['Flight Distance'] = combined_airline['Flight Distance'].apply(np.log)
combined_airline

#%% Ordinal encoding object columns
#Ordinally encode the object columns in a copy of the combined airline dataset
object_cols = [col for col in combined_airline.columns if combined_airline[col].dtype == 'object']
ordinal_encoder = OrdinalEncoder()
combined_airline_copy = combined_airline.copy()
combined_airline_copy[object_cols] = ordinal_encoder.fit_transform(combined_airline_copy[object_cols])
combined_airline_copy

#%% Simple imputation

#Perform simple imputation to create a new training and testing "x" dataset

#Simple imputation with median
y = combined_airline_copy['satisfaction']
x = combined_airline_copy.drop(columns='satisfaction')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state = 0)

simple_imputer = SimpleImputer(strategy='median')
imputed_x_train = pd.DataFrame(simple_imputer.fit_transform(x_train))
imputed_x_test = pd.DataFrame(simple_imputer.transform(x_test))

#imputation removes column names
imputed_x_train.columns = x_train.columns
imputed_x_test.columns = x_test.columns

#%% Random Forest model

#Creates a simple RF model and determines feature importance - yet to put in parameter tuning

#Random Forest model creation
model = RandomForestClassifier()
model.fit(imputed_x_train, y_train)
predictions = model.predict(imputed_x_test)
mae = mean_absolute_error(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
cf_matrix = confusion_matrix(y_test, predictions)

#Creates feature importances and creates a graph
feature_importances = model.feature_importances_
print(imputed_x_train.columns)
importance_df = pd.DataFrame({'Feature Names': imputed_x_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
importance_plot = importance_df.plot(kind='barh', x='Feature Names', y='Importance', legend=False, figsize=(15, 10))
plt.title(f'Feature Importances in Random Forest, with accuracy {accuracy*100:.2f}%')
for i, v in enumerate(importance_df['Importance']):
    importance_plot.text(v + 0.0, i, f'{v:.4f}', va='center')

#%% XGBoost functionality w/ train test validation split

#Run XGBoost model to perform enseble learning

#Train-test-validation split using train test split
X_train, X_val, Y_train, Y_val = train_test_split(imputed_x_train, y_train, test_size=0.25)

#Creation of the XGB model w/ n-estimators = 500 and learning rate of 0.05, with early stopping rounds for gradient descent = 5
xg_model = XGBClassifier(n_estimators=500, learning_rate=0.05, early_stopping_rounds=5)
xg_model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
xg_predictions = xg_model.predict(x_test)
xg_mae = mean_absolute_error(y_test, xg_predictions)
xg_accuracy = accuracy_score(y_test, xg_predictions)
xg_cf_matrix = confusion_matrix(y_test, xg_predictions)
xg_class_report = classification_report(y_test, xg_predictions)

feature_importances = xg_model.feature_importances_
xg_importance_df = pd.DataFrame({'Feature Names': imputed_x_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
xg_importance_plot = xg_importance_df.plot(kind='barh', x='Feature Names', y='Importance', legend=False, figsize=(15, 10))
plt.title(f'Feature Importances in XGBoost, with Accuracy {xg_accuracy*100:.2f}% and MAE {xg_mae:.2f}')
for i, v in enumerate(xg_importance_df['Importance']):
    xg_importance_plot.text(v + 0.0, i, f'{v:.4f}', va='center')

#Prints relevant information
print(xg_class_report)
print(xg_mae)
print(xg_accuracy)
print(xg_cf_matrix)
