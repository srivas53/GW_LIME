# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:46:54 2023

@author: vsriva11
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import geopandas as gpd
import libpysal as ps
from libpysal  import weights
from libpysal.weights import Queen, KNN
import libpysal.weights as wts
import esda
from esda.moran import Moran, Moran_Local
from giddy.directional import Rose
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import sklearn
from sklearn.preprocessing import StandardScaler
import pyproj
import utm
from pyHSICLasso import HSICLasso
import json
from shapely.geometry import MultiPolygon, Polygon
from statistics import stdev
import ast
from sklearn.linear_model import Ridge, lars_path, LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import statsmodels.api as sm
from statsmodels.api import WLS
from sklearn.svm import SVR
from scipy import stats
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME\XAI")


################Calculate optimal bandwidth for GWR########################################

prepare_GWR_weights = False #Set to True to recalculate optimal bandwidth

if prepare_GWR_weights == True:
    mex_migration = pd.read_csv("./data/mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df.csv")
    
    
    ##Add target variable (predictions from the GraphSAGE model, used as the ground truth value, refer to Section 3.3.3, paragraph 1 of the paper)
    with open("./data/predictions_target_with_crime_no_implicit_target_num_migrants.txt", 'r') as file:
        predictions = ast.literal_eval(file.read())
        
    
    
    target = 'sum_num_initmig' 
    predictions = pd.DataFrame(predictions, columns = [target])
    
    mex_migration = pd.merge(mex_migration, predictions, left_index = True, right_index = True, how = 'left')
    
    
    
    #Execute Principal Component Extraction to avoid issues due to multicollinearity while calculating optimal bandwidth for GWR (See Section 3.3.2 of the paper)
    all_columns = list(mex_migration.columns) #store a list of all columns in the dataset
    predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', target, 'Latitude', 'Longitude']] #exclude columns that are not in the expalanatory variable set
    predictor_dataframe = mex_migration[predictor_columns]
    
    
    predictor_dataframe = scaler.fit_transform(predictor_dataframe)
    
    
    #cross validation to select optimal PC's to retain was executed, 20 to retain was the ideal value
    
    n_components = 20
    
    pca = PCA(n_components=n_components)
    
    # Fit and transform the scaled data using PCA
    principal_components = pca.fit_transform(predictor_dataframe)
    
    # #Convert to a dataframe for easier analysis
    
    column_names_pcs = []
    
    for i in range(n_components):
        column_names_pcs.append('PC_' + str(i+1))
    
    
    
    principal_components_df = pd.DataFrame(principal_components, columns = column_names_pcs)
    
    mex_dataframe_principal_components = pd.merge(mex_migration[['GEO2_MX', target, 'Latitude', 'Longitude']], principal_components_df, left_index = True, right_index = True, how = 'left')
    
    u = mex_dataframe_principal_components['Longitude']
    v = mex_dataframe_principal_components['Latitude']
    coords = list(zip(u,v))
    
    #prepare data in the requred format for extracting neighborhood weights
    
    all_columns = list(mex_dataframe_principal_components.columns) #store a list of all columns in the dataset
    
    predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', target, 'Latitude', 'Longitude']]
    predictor_dataframe = mex_dataframe_principal_components[predictor_columns]
    
    #scale the data
    #predictor_dataframe = scaler.fit_transform(predictor_dataframe)
    
    
    #X = predictor_dataframe.copy()
    X = predictor_dataframe.values
    
    
    predicted_dataframe = mex_dataframe_principal_components[[target]] 
    
    
    #change dtypes for compatibility
    y = predicted_dataframe.values.reshape((-1,1)) # reshape is needed to have column array
    
    gwr_selector = Sel_BW(coords, y, X, spherical= True)
    gwr_bw = gwr_selector.search()
    print('GWR bandwidth =', gwr_bw)
    #Optimal GWR bandwidth = 151.0
    
    
    
    #Fit a GWR model to extract the weights (adaptive bisquare, golden selection rule, refer to Section 3.3.2)
    gwr_results = GWR(np.array(coords), np.array(y), np.array(X), gwr_bw).fit()
    
    
    weights_gwr = gwr_results.W
    
    weights_array = []
    
    for item in weights_gwr:
        weights_array.append(list(item))
        
    
    gwr_weights_df = pd.DataFrame(columns = ['GEO2_MX', 'weights'])
    
    
    for i in range(len(mex_dataframe_principal_components)):
        gwr_weights_df.loc[i, 'GEO2_MX' ] = mex_dataframe_principal_components.loc[i, 'GEO2_MX']
        gwr_weights_df.loc[i, 'weights' ] = weights_array[i]
    
    #export weights per municipality for use later in the XAI computations
    
    gwr_weights_df.to_csv('./data/GWR_Weights_adaptive_bisquare_target_num_migrants_crime_variables_no_implicit_PCA.csv', index= False)


##Execute XAI computations (LIME and GW-LIME)


#Import dataset

mex_migration = pd.read_csv("./data/mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df.csv")


#Add target variable (predictions from the GraphSAGE model, used as the ground truth value, refer to Section 3.3.3, paragraph 1 of the paper)
with open("./data/predictions_target_with_crime_no_implicit_target_num_migrants.txt", 'r') as file:
    predictions = ast.literal_eval(file.read())
    


target = 'sum_num_initmig' 
predictions = pd.DataFrame(predictions, columns = [target])

mex_migration = pd.merge(mex_migration, predictions, left_index = True, right_index = True, how = 'left')




#######Model 1: GW-LIME###########

#Data cleaning and preprocessing
mex_migration_GWR_LIME = mex_migration.copy()

mex_migration_GWR_LIME['GEO2_MX'] = mex_migration_GWR_LIME['GEO2_MX'].astype(str)

mex_migration_GWR_LIME = mex_migration_GWR_LIME.fillna(0)


#Prepare a dataframe containing only the predictor variables
all_columns = list(mex_migration_GWR_LIME.columns)
predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', target, 'Latitude', 'Longitude']]

predictor_dataframe = mex_migration_GWR_LIME[predictor_columns]


#make a copy to use for correlation analysis later in the pipeline
predictor_dataframe_for_correlation = predictor_dataframe.copy()



#standardize the predictor variable dataframe

predictor_dataframe = scaler.fit_transform(predictor_dataframe)

X = predictor_dataframe.copy()

predicted_dataframe = mex_migration_GWR_LIME[[target]] 

y = predicted_dataframe.values.reshape((-1,1)) # reshape is needed to have column array

#import neighborhood weights calculated using GWR

gwr_weights_df = pd.read_csv("./data/GWR_Weights_adaptive_bisquare_target_num_migrants_crime_variables_no_implicit_PCA.csv")

gwr_weights_df['weights'] = gwr_weights_df['weights'].apply(ast.literal_eval)

gwr_weights_df['GEO2_MX'] = gwr_weights_df['GEO2_MX'].astype(str)

##############################Note: Not relevant for GW-LIME: correlation analysis##############
# correlation_matrix = predictor_dataframe_for_correlation.corr()

# #for a given variable of interest, return any other variables that have >= (+-).90 coreelation

# def return_other_correlated_variables(instance_of_interest, feature_name):
#     neighborhood = gwr_weights_df[gwr_weights_df['GEO2_MX']== instance_of_interest]
#     neighborhood = neighborhood.reset_index(drop = True)
#     neighborhood_wts = neighborhood.loc[0, 'weights']
#     indexes_to_consider = []
#     for i in range(len(neighborhood_wts)):
#         if neighborhood_wts[i] > 0.50:
#             indexes_to_consider.append(i)
    
#     subset_neighborhood = predictor_dataframe_for_correlation.iloc[indexes_to_consider]
#     neighborhood_correlation_matrix = subset_neighborhood.corr()
#     corr_subset = pd.DataFrame(neighborhood_correlation_matrix[feature_name])
#     corr_subset = corr_subset.reset_index()
    
#     corr_dict = {}
    
#     for i in range(len(corr_subset)):
#         if abs(corr_subset.loc[i, feature_name])>=0.80:
#             if corr_subset.loc[i, 'index'] != feature_name:
#                 corr_dict[corr_subset.loc[i, 'index']] = corr_subset.loc[i, feature_name]
    
#     return corr_dict

                
# corr_value =  return_other_correlated_variables('484001001', 'perc_piped_inside_dwelling_watsup')



#Define function to execute K-Lasso, and return the given values: k most important features, local R squared, local MAE, local MAPE (Refer to Section 3.2, paragraph 5 for more on K-Lasso, and Section 3.3.3, paragraph 1 for local R-sqared and MAE) 
def gwr_lasso_explain_instance(geo_id_of_interest, number_of_features_k, return_only_performance_metrics = True): #provide id as a string
    instance_of_interest = gwr_weights_df[gwr_weights_df['GEO2_MX'] ==geo_id_of_interest]
    instance_index = instance_of_interest.index
    instance_of_interest = instance_of_interest.reset_index(drop = True)
    
    #processing weights according to Wheeler's implementation
    instance_of_interest_neighborhood_wts = instance_of_interest.loc[0, 'weights']
    instance_of_interest_neighborhood_wts = np.array(instance_of_interest_neighborhood_wts)
    instance_of_interest_neighborhood_wts = instance_of_interest_neighborhood_wts.reshape(-1, 1)
    instance_of_interest_neighborhood_wts = np.sqrt(instance_of_interest_neighborhood_wts)
    weighted_X = instance_of_interest_neighborhood_wts * X
    weighted_Y = instance_of_interest_neighborhood_wts * y
    weighted_Y = np.ravel(weighted_Y)
    
    #call lars and save solutions
    
    alphas, _, coefs = lars_path(weighted_X,
                                 weighted_Y,
                                 method='lasso',
                                 verbose=False)
    
    #number_of_features_k = 10
    for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= number_of_features_k:
            break

    used_features = nonzero
    

    ##########################using sklearn to fit LM###############
    # model_regressor = LinearRegression(fit_intercept=True)
    
    # easy_model = model_regressor
    
    # easy_model.fit(X[:, used_features],
    #             y, sample_weight=np.ravel(instance_of_interest_neighborhood_wts))
    
    # # easy_model.fit(weighted_X[:, used_features],
    # #             weighted_Y_copy)
        
    # #calculate adjusted r squared
    # prediction_score = easy_model.score(
    # X[:, used_features],
    # y, sample_weight=np.ravel(instance_of_interest_neighborhood_wts))
    
    # # Calculate the number of observations for which predictions were calculated
    # n = len(y)
    # p = number_of_features_k
    
    # adjusted_r_squared = 1 - ((1 - prediction_score) * (n - 1) / (n - p - 1))
    
    # #caclulate RMSE and also store error values
    
    # predicted_values = easy_model.predict(X[:, used_features])
    
    # #consider neighborhood wts for calculating RMSE
    # residuals = y - predicted_values
    
    # #multiply residuals by neighborhood wts
    # weighted_residuals = residuals * instance_of_interest_neighborhood_wts
    
    # rmse = np.sqrt(np.mean((weighted_residuals)**2))
    # mae = np.mean(np.abs(weighted_residuals))
    
    # #convert residuals to DF for futher analysis later
    # instance_residual = residuals[instance_index][0][0]
    # residuals = pd.DataFrame(residuals, columns = ['residuals'])
    # weighted_residuals = pd.DataFrame(weighted_residuals, columns = ['weighted_residuals']) 
    
    
    #########using statsmodels to fit LM############################
    X_subset = X[:, used_features]
    X_subset = sm.add_constant(X_subset) # manually add a constant here
    
    model_regressor = WLS(y, X_subset, weights=np.ravel(instance_of_interest_neighborhood_wts))
    results = model_regressor.fit()
    beta_coefficients = results.params #first one is beta_0
    predictions = results.predict(X_subset)
    
    #get adjusted r squared
    adjusted_r_squared = results.rsquared_adj
    

    
    #return p-value significance of each variable
    significance_values = results.pvalues
    residuals = results.resid #unweighted residuals
    residuals = residuals.reshape(-1, 1)
    weighted_residuals = residuals * instance_of_interest_neighborhood_wts
    
    
    #weighted_residuals = residuals * instance_of_interest_neighborhood_wts
    instance_residual = weighted_residuals[instance_index][0][0]
    mae = np.mean(np.abs(weighted_residuals))
    
    #MAPE
    #residuals_for_mape = results.resid #in the array form required for MAPE calculation
    
    #index 1740 is problematic since its actual value is 0, so convert its weight to 0 to have consisten results
    #instance_of_interest_neighborhood_wts[1740] = 0
    mape = np.mean(np.abs((residuals) / (y+ 1)) * instance_of_interest_neighborhood_wts) * 100 #y+1 to avoid 0 division

    
    
    instance_predicted_value = predictions[instance_index]
    
    
    
    #get rmse
    mse = np.mean(weighted_residuals ** 2)

    ##Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)    
    
    
    ############################
    
    
    if return_only_performance_metrics == False:
        explanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'p-value'])
        
        for i in range(len(used_features)):
            feature_index = used_features[i]
            predictor_name = predictor_columns[feature_index]
            
            #for sklearn
            # beta_coefficient = easy_model.coef_[0][i]
            # row = {'feature_name' : predictor_name, 'beta_coefficient' : beta_coefficient, 'p-value': 'N/A'}
            #explanations_df = pd.concat([explanations_df, pd.DataFrame([row])], ignore_index=True)
            
            
            #for statsmodels
            beta_coefficient = beta_coefficients[i+1]
            row = {'feature_name' : predictor_name, 'beta_coefficient' : beta_coefficient, 'p-value': significance_values[i+1]}
            #explanations_df = explanations_df.append(row, ignore_index=True)
            explanations_df = pd.concat([explanations_df, pd.DataFrame([row])], ignore_index=True)

        
        explanations_df = explanations_df.reindex(explanations_df['beta_coefficient'].abs().sort_values(ascending=False).index)

    else:
        explanations_df = 'Null'
        
    return [explanations_df, adjusted_r_squared, rmse, mae, instance_residual, mape, residuals, weighted_residuals, instance_predicted_value]


#####Execute function defined above to extract explanations and performance metrics for a given municipality#########

explanation_yucatan = gwr_lasso_explain_instance('484004002', 10, return_only_performance_metrics = True)[1:4]
explanation_sinaloa = gwr_lasso_explain_instance('484010034', 10, return_only_performance_metrics = False)[0]#484025003
explanation_colima  = gwr_lasso_explain_instance('484016026', 10, return_only_performance_metrics = False)[0]#484025003



##For every municipality, return performance metrics values for local fidelity experiment (Refer Section 3.3.3, paragraph 1)

os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME_VADER\XAI\data\Explanations\GW_LIME")

list_of_munis = list(mex_migration_GWR_LIME['GEO2_MX'])

performance_indicatiors_GWR_LIME = pd.DataFrame(columns = ['GEO2_MX', 'adjusted_r_squared', 'rmse', 'mae', 'residual', 'mape'])


for item in list_of_munis:
    [local_r_squared, rmse, mae, residuals, mape] = gwr_lasso_explain_instance(item, 10, return_only_performance_metrics = True)[1:6]
    #rmse = gwr_lasso_explain_instance(item, 10, return_only_performance_metrics = True)[2]
    row = {'GEO2_MX' : item, 'adjusted_r_squared' : local_r_squared,'rmse' : rmse, 'mae' : mae, 'residual': residuals, 'mape': mape}
    #performance_indicatiors_GWR_LIME = performance_indicatiors_GWR_LIME.append(row, ignore_index=True)
    performance_indicatiors_GWR_LIME = pd.concat([performance_indicatiors_GWR_LIME, pd.DataFrame([row])], ignore_index=True)
    

performance_indicatiors_GWR_LIME.to_csv("all_performance_indicators_GWR_LIME.csv", index = False)

#######

#Extract residuals for a sample municipality for analyzing residual distribution

# [sample_residuals_GWR_Lasso, sample_weighted_residuals_GWR_Lasso] = gwr_lasso_explain_instance('484014045', 10)[5:]

# # sample_residuals_GWR_Lasso.to_csv('sample_residuals_GWR_Lasso_484014045.csv', index = False)

# # sample_weighted_residuals_GWR_Lasso.to_csv('sample_weighted_residuals_GWR_Lasso_484014045.csv', index = False)


# #sample_residuals_GWR_Lasso = pd.read_csv()

# #Make QQ plot to check distribution


# plt.figure(figsize=(6, 6))
# stats.probplot(sample_residuals_GWR_Lasso['residuals'], dist="norm", plot=plt)
# plt.title("Q-Q Plot for residuals - GWR_LIME")
# plt.xlabel("Theoretical Quantiles")
# plt.ylabel("Sample Quantiles")
# plt.show()

# plt.figure(figsize=(6, 6))
# stats.probplot(sample_weighted_residuals_GWR_Lasso['weighted_residuals'], dist="norm", plot=plt)
# plt.title("Q-Q Plot for weighted residuals - GWR_LIME")
# plt.xlabel("Theoretical Quantiles")
# plt.ylabel("Sample Quantiles")
# plt.show()



#Local Fidelity Experiment: Calculate median values of performance metrics (Refer to Section 3.3.3, paragraph 1 for further details)

summary = performance_indicatiors_GWR_LIME['adjusted_r_squared'].describe()
median = performance_indicatiors_GWR_LIME['adjusted_r_squared'].median()
percentiles = np.percentile(performance_indicatiors_GWR_LIME['adjusted_r_squared'], [25, 50, 75], axis=0)
mean = performance_indicatiors_GWR_LIME['adjusted_r_squared'].mean()
std_dev = stdev(performance_indicatiors_GWR_LIME['adjusted_r_squared'])

#Create a summary DataFrame for adjusted R-Squared

summary_df_adj_rsquared = pd.DataFrame({
    'Mean': [mean],
    'Std Dev': [std_dev],
    '25th Percentile': [percentiles[0]],
    'Median': [median],
    '75th Percentile': [percentiles[2]],
    'Max' : max(performance_indicatiors_GWR_LIME['adjusted_r_squared']),
    'Min' : min(performance_indicatiors_GWR_LIME['adjusted_r_squared'])
}, index=['Summary'])

#Create a summary DataFrame for adjusted RMSE

summary_rmse = performance_indicatiors_GWR_LIME['rmse'].describe()
median_rmse = performance_indicatiors_GWR_LIME['rmse'].median()
percentiles_rmse = np.percentile(performance_indicatiors_GWR_LIME['rmse'], [25, 50, 75], axis=0)
mean_rmse = performance_indicatiors_GWR_LIME['rmse'].mean()
std_dev_rmse = stdev(performance_indicatiors_GWR_LIME['rmse'])

# Create a summary DataFrame
summary_df_rmse = pd.DataFrame({
    'Mean': [mean_rmse],
    'Std Dev': [std_dev_rmse],
    '25th Percentile': [percentiles_rmse[0]],
    'Median': [median_rmse],
    '75th Percentile': [percentiles_rmse[2]],
    'Max' : max(performance_indicatiors_GWR_LIME['rmse']),
    'Min' : min(performance_indicatiors_GWR_LIME['rmse'])
}, index=['Summary'])

#Create a summary DataFrame for adjusted MAE

summary_mae = performance_indicatiors_GWR_LIME['mae'].describe()
median_mae = performance_indicatiors_GWR_LIME['mae'].median()
percentiles_mae = np.percentile(performance_indicatiors_GWR_LIME['mae'], [25, 50, 75], axis=0)
mean_mae = performance_indicatiors_GWR_LIME['mae'].mean()
std_dev_mae = stdev(performance_indicatiors_GWR_LIME['mae'])

# Create a summary DataFrame
summary_df_mae = pd.DataFrame({
    'Mean': [mean_mae],
    'Std Dev': [std_dev_mae],
    '25th Percentile': [percentiles_mae[0]],
    'Median': [median_mae],
    '75th Percentile': [percentiles_mae[2]],
    'Max' : max(performance_indicatiors_GWR_LIME['mae']),
    'Min' : min(performance_indicatiors_GWR_LIME['mae'])
}, index=['Summary'])

#Create a summary DataFrame for adjusted MAPE

summary_mape = performance_indicatiors_GWR_LIME['mape'].describe()
median_mape = performance_indicatiors_GWR_LIME['mape'].median()
percentiles_mape = np.percentile(performance_indicatiors_GWR_LIME['mape'], [25, 50, 75], axis=0)
mean_mape = performance_indicatiors_GWR_LIME['mape'].mean()
std_dev_mape = stdev(performance_indicatiors_GWR_LIME['mape'])

# Create a summary DataFrame
summary_df_mape = pd.DataFrame({
    'Mean': [mean_mape],
    'Std Dev': [std_dev_mape],
    '25th Percentile': [percentiles_mape[0]],
    'Median': [median_mape],
    '75th Percentile': [percentiles_mape[2]],
    'Max' : max(performance_indicatiors_GWR_LIME['mape']),
    'Min' : min(performance_indicatiors_GWR_LIME['mape'])
}, index=['Summary'])



summary_df_adj_rsquared.to_csv('adj_r_squared_performance_metrics_summary_GWR_LIME.csv', index = False)
summary_df_rmse.to_csv('rmse_performance_metrics_summary_GWR_LIME.csv', index = False)
summary_df_mae.to_csv('mae_performance_metrics_summary_GWR_LIME.csv', index = False)
summary_df_mape.to_csv('mape_performance_metrics_summary_GWR_LIME.csv', index = False)


#For each municipality, extract top 10 most important features for critical inferential tasks (See Section 4)
os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME_VADER\XAI\data\Explanations\GW_LIME\top_k_important_features")

list_of_munis = list(mex_migration_GWR_LIME['GEO2_MX'])

for item in list_of_munis:
    explanation= gwr_lasso_explain_instance(item, 10, return_only_performance_metrics = False)[0]
    #count number of satellite PC's, do k = k+n, where n= number of satellite PC's
    important_features = list(explanation['feature_name'])
    # count_satellite = 0
    # for element in important_features:
    #     if 'Satellite' in element:
    #         count_satellite+=1
    # if count_satellite!=0:
    #     explanation= gwr_lasso_explain_instance(item, 10 + count_satellite, return_only_performance_metrics = False)[0]
    explanation.to_csv(f'GWR_LIME_explanation_{item}.csv', index = False)




###Model 2: OLS LIME

#Import LIME package
import lime
import lime.lime_tabular


#Prepare dataset for processing
mex_migration_OLS_LIME = mex_migration.copy()

mex_migration_OLS_LIME['GEO2_MX'] = mex_migration_OLS_LIME['GEO2_MX'].astype(str)

mex_migration_OLS_LIME = mex_migration_OLS_LIME.fillna(0)

all_columns = list(mex_migration_OLS_LIME.columns)

predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', 'Latitude', 'Longitude', target]]

predictor_dataframe = mex_migration_OLS_LIME[predictor_columns]

predicted_dataframe = mex_migration_OLS_LIME[['GEO2_MX', target]]
predicted_dataframe['GEO2_MX'] = predicted_dataframe['GEO2_MX'].astype(str)


#define predict function needed for LIME package

def predict_migration(list_ids):
    list_of_predictions = []
    for i in range(len(list_ids)):
        pred_value = predicted_dataframe.loc[predicted_dataframe['GEO2_MX'] == list_ids[i], target].iloc[0]
        list_of_predictions.append(pred_value)
    return np.array(list_of_predictions)



#initiate the LIME Explainer instance for a tabular dataset with regression mode (See LIME documentation from the original authors for more information: https://github.com/marcotcr/lime )
training_dataset = np.array(predictor_dataframe)

explainer = lime.lime_tabular.LimeTabularExplainer(training_dataset, feature_names= predictor_columns, discretize_continuous=False, 
                                                  mode = 'regression', feature_selection = 'lasso_path')


#Define a function to generate expalations for a given municipality
def OLS_LIME_Instance_Explainer(instance_to_explain, inverse_array, no_of_features):
    exp = explainer.explain_instance(instance_to_explain, predict_fn = predict_migration, num_features=no_of_features, num_samples = len(training_dataset), list_of_prediction_id = inverse_array)
    #print(exp.as_list())
    
    return([exp.as_list(), exp.score, list(exp.weights), exp.predicted_values])


#Define a function to select the instance of interest

def OLS_LIME_Instance_Selector(explanatory_variables_df, explained_variable_df, instance_to_explain, number_of_features_k):
    #random_sample_of_interest = predicted_dataframe['GEO2_MX'].sample()
    #id_of_interest = random_sample_of_interest.values[0]
    id_of_interest = instance_to_explain
    #id_of_interest_index = random_sample_of_interest.index[0]
    id_of_interest_index = explained_variable_df[explained_variable_df['GEO2_MX'] == id_of_interest].index[0]
    instance_selected = explanatory_variables_df.iloc[id_of_interest_index, :] #Replace with automated extraction of the row
    instance_selected = np.array(instance_selected)
    
    #prepare inverse array
    inverse_array_selected = list(explained_variable_df['GEO2_MX'])
    
    #Remove the id for the instance of interest, and move it to the top of the array
    inverse_array_selected.remove(id_of_interest) 
    inverse_array_selected.insert(0, id_of_interest)
    LIME_metrics = OLS_LIME_Instance_Explainer(instance_selected, inverse_array_selected, number_of_features_k)
    return [LIME_metrics, inverse_array_selected]




#Define a function to extract performance metrics and top-k most important features (similar to the gwr_lime_expalin_instance function)
 
def OLS_LIME_Explainer(explanatory_df, target_variable_df, instance_of_interest, number_of_features_k, return_only_performance_metrics = True, return_only_weights = False):
    [ols_results, original_order_for_rmse]  = OLS_LIME_Instance_Selector(explanatory_df, target_variable_df, instance_of_interest, number_of_features_k)
    r_squared = ols_results[1]
    
    #adjusted r-squared
    n = len(target_variable_df)
    p = number_of_features_k
    
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    
    #rmse
    predicted_values = ols_results[3]
    neighborhood_weights = ols_results[2] #first index is of the explained instance
    original_order_for_rmse = pd.DataFrame(original_order_for_rmse, columns = ['GEO2_MX'])

    #####weights df to be used downstream
    neighborhood_weights_df = pd.DataFrame(neighborhood_weights, columns = ['feature weights'])
    neighborhood_weights_df = pd.merge(original_order_for_rmse, neighborhood_weights_df, left_index = True, right_index = True, how = 'left')
    
        


    if return_only_weights == False:
        neighborhood_weights = np.array(neighborhood_weights)
        neighborhood_weights = neighborhood_weights.reshape(-1, 1)
        original_labels_for_rmse =  pd.merge(original_order_for_rmse, target_variable_df, on = 'GEO2_MX', how = 'left')
        original_labels_for_rmse = np.array(original_labels_for_rmse[target])
        
        
        residuals = original_labels_for_rmse - predicted_values
        instance_residual = residuals[0]
        residuals = residuals.reshape(-1, 1)
        weighted_residuals = residuals * neighborhood_weights
        
        rmse = np.sqrt(np.mean((weighted_residuals)**2))
        mae = np.mean(np.abs(weighted_residuals))
        
        #convert residuals to DF for futher analysis later
        residuals_df = pd.DataFrame(residuals, columns = ['residuals'])
        weighted_residuals = pd.DataFrame(weighted_residuals, columns = ['weighted_residuals']) 
        
        #MAPE
        #484024009: set its weight to 0, refer to GWR to see why we do it 
        neighborhood_weights = np.array(neighborhood_weights_df['feature weights'])
        neighborhood_weights = neighborhood_weights.reshape(-1, 1)
        original_labels_for_rmse = original_labels_for_rmse.reshape(-1, 1)
        
        mape = np.mean(np.abs((residuals) / (original_labels_for_rmse + 1)) * neighborhood_weights) * 100 #y+1 to avoid 0 division

    
    else:
        [weighted_residuals, instance_residual, residuals, mae , adjusted_r_squared , rmse] = [None, None, None, None, None, None]
        
    
    if return_only_performance_metrics == False:
        explanation_df = pd.DataFrame(ols_results[0], columns = ['feature_name', 'beta_coefficient'] )
    else: 
        explanation_df = 'None'
    
    return [adjusted_r_squared, rmse, mae, instance_residual, mape, explanation_df, residuals_df, weighted_residuals, neighborhood_weights_df]
        
    
#Generate LIME explanations for a given municipality

ols_results_oaxaca  = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, '484020475', 10, return_only_performance_metrics = False)[4]
ols_results_aguacalientes  = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, '484001002', 10, return_only_performance_metrics = False)[4]



#Store performance metrics values for every municipality (TBC 01/10/2024)
os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME_VADER\XAI\data\Explanations\OLS_LIME")


list_of_munis = list(mex_migration_OLS_LIME['GEO2_MX'])


performance_indicatiors_OLS_LIME = pd.DataFrame(columns = ['GEO2_MX', 'adjusted_r_squared', 'rmse', 'mae', 'residual', 'mape'])


for item in list_of_munis:
    [local_r_squared, rmse, mae, residual, mape] = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, item, 10, return_only_performance_metrics = True ) [0:5]
    row = {'GEO2_MX' : item, 'adjusted_r_squared' : local_r_squared,'rmse' : rmse, 'mae': mae, 'residual': residual, 'mape' : mape }
    performance_indicatiors_OLS_LIME = pd.concat([performance_indicatiors_OLS_LIME, pd.DataFrame([row])], ignore_index=True)
    
##execute 10/25#####

performance_indicatiors_OLS_LIME.to_csv("all_performance_indicators_OLS_LIME.csv", index = False)

####

#performance_indicatiors_OLS_LIME = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants\all_performance_indicators_OLS_LIME.csv")
###Descriptive statistics for local r squared

summary = performance_indicatiors_OLS_LIME['adjusted_r_squared'].describe()
median = performance_indicatiors_OLS_LIME['adjusted_r_squared'].median()
percentiles = np.percentile(performance_indicatiors_OLS_LIME['adjusted_r_squared'], [25, 50, 75], axis=0)
mean = performance_indicatiors_OLS_LIME['adjusted_r_squared'].mean()
std_dev = stdev(performance_indicatiors_OLS_LIME['adjusted_r_squared'])

# Create a summary DataFrame
summary_df_adj_rsquared_OLS_LIME = pd.DataFrame({
    'Mean': [mean],
    'Std Dev': [std_dev],
    '25th Percentile': [percentiles[0]],
    'Median': [median],
    '75th Percentile': [percentiles[2]],
    'Max' : max(performance_indicatiors_OLS_LIME['adjusted_r_squared']),
    'Min' : min(performance_indicatiors_OLS_LIME['adjusted_r_squared'])
}, index=['Summary'])

###Descriptive statistics for rmse

summary_rmse = performance_indicatiors_OLS_LIME['rmse'].describe()
median_rmse = performance_indicatiors_OLS_LIME['rmse'].median()
percentiles_rmse = np.percentile(performance_indicatiors_OLS_LIME['rmse'], [25, 50, 75], axis=0)
mean_rmse = performance_indicatiors_OLS_LIME['rmse'].mean()
std_dev_rmse = stdev(performance_indicatiors_OLS_LIME['rmse'])

# Create a summary DataFrame
summary_df_rmse_OLS_LIME = pd.DataFrame({
    'Mean': [mean_rmse],
    'Std Dev': [std_dev_rmse],
    '25th Percentile': [percentiles_rmse[0]],
    'Median': [median_rmse],
    '75th Percentile': [percentiles_rmse[2]],
    'Max' : max(performance_indicatiors_OLS_LIME['rmse']),
    'Min' : min(performance_indicatiors_OLS_LIME['rmse'])
}, index=['Summary'])



summary_mae = performance_indicatiors_OLS_LIME['mae'].describe()
median_mae = performance_indicatiors_OLS_LIME['mae'].median()
percentiles_mae = np.percentile(performance_indicatiors_OLS_LIME['mae'], [25, 50, 75], axis=0)
mean_mae = performance_indicatiors_OLS_LIME['mae'].mean()
std_dev_mae = stdev(performance_indicatiors_OLS_LIME['mae'])


summary_df_mae_OLS_LIME = pd.DataFrame({
    'Mean': [mean_mae],
    'Std Dev': [std_dev_mae],
    '25th Percentile': [percentiles_mae[0]],
    'Median': [median_mae],
    '75th Percentile': [percentiles_mae[2]],
    'Max' : max(performance_indicatiors_OLS_LIME['mae']),
    'Min' : min(performance_indicatiors_OLS_LIME['mae'])
}, index=['Summary'])

summary_mape = performance_indicatiors_OLS_LIME['mape'].describe()
median_mape = performance_indicatiors_OLS_LIME['mape'].median()
percentiles_mape = np.percentile(performance_indicatiors_OLS_LIME['mape'], [25, 50, 75], axis=0)
mean_mape = performance_indicatiors_OLS_LIME['mape'].mean()
std_dev_mape = stdev(performance_indicatiors_OLS_LIME['mape'])


summary_df_mape_OLS_LIME = pd.DataFrame({
    'Mean': [mean_mape],
    'Std Dev': [std_dev_mape],
    '25th Percentile': [percentiles_mape[0]],
    'Median': [median_mape],
    '75th Percentile': [percentiles_mape[2]],
    'Max' : max(performance_indicatiors_OLS_LIME['mape']),
    'Min' : min(performance_indicatiors_OLS_LIME['mape'])
}, index=['Summary'])




os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants")


summary_df_adj_rsquared_OLS_LIME.to_csv('adj_r_squared_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_rmse_OLS_LIME.to_csv('rmse_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_mae_OLS_LIME.to_csv('mae_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_mape_OLS_LIME.to_csv('mape_performance_metrics_summary_OLS_LIME.csv', index = False)


###Export top 10 explnations for OLS

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants\top_k_important_features\Original_top_10_features\OLS_LIME")
list_of_munis = list(mex_migration_OLS_LIME['GEO2_MX'])


#ols_results_moderate = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, '484014045', 10, return_only_performance_metrics = False)[4]
for item in list_of_munis:
    explanation= OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, item, 10, return_only_performance_metrics = False)[4]
    #count number of satellite PC's, do k = k+n, where n= number of satellite PC's
    # if count_satellite!=0:
    #     explanation= OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, item, 10 + count_satellite, return_only_performance_metrics = False)[4]

    explanation.to_csv(f'OLS_LIME_explanation_{item}.csv', index = False)








#####residual -- I (global Moran's I)######################
#import residuals for GWR LIMe

gwr_lime_residuals = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants\all_performance_indicators_GWR_LIME.csv")

gwr_lime_residuals = gwr_lime_residuals[['GEO2_MX', 'residual']]

#prepare weights matrix using KNN

centroid_coordinates = mex_migration[['Longitude', 'Latitude']]

centroid_coordinates = np.array(centroid_coordinates)


w_knn = KNN(centroid_coordinates, k=5)


gwr_lime_residuals_array = np.array(gwr_lime_residuals['residual'])


moran = Moran(gwr_lime_residuals_array, w_knn)



print("Moran's I:", moran.I)
print("Expected Moran's I:", moran.EI)
print("p-value:", moran.p_norm)
print("z-score:", moran.z_norm)


ols_lime_residuals = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants\all_performance_indicators_OLS_LIME.csv")

ols_lime_residuals = ols_lime_residuals[['GEO2_MX', 'residual']]

ols_lime_residuals_array = np.array(ols_lime_residuals['residual'])

moran_2 = Moran(ols_lime_residuals_array, w_knn)



print("Moran's I:", moran_2.I)
print("Expected Moran's I:", moran_2.EI)
print("p-value:", moran_2.p_norm)
print("z-score:", moran_2.z_norm)









############To figure out why OLS LIME is better than GWR LIME, hypothseis is feature space and geo-space are more or less similar


###################GraphLIME###############################

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\n_hop_neighbors_subset\three_hop")

municipality_n_hop_neighbors = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\mexico_migration_three_hop_neighbors.csv")


#pyHSIC Lasso needs input in the form of CSV, and we define a function that subsets data according to 2-hop structure in form of csv
#This dataset also requires target label to be names as 'class'

def export_subset_for_HSIC(instance_of_interest):
    n_hop_neighborhood_of_interest = municipality_n_hop_neighbors[municipality_n_hop_neighbors['GEO2_MX'] ==instance_of_interest]
    n_hop_neighborhood_of_interest = n_hop_neighborhood_of_interest.reset_index(drop = True)
    n_hop_neighborhood_of_interest = n_hop_neighborhood_of_interest.loc[0, 'adjacent_municipalities']
    n_hop_neighborhood_of_interest = ast.literal_eval(n_hop_neighborhood_of_interest)
    n_hop_neighborhood_of_interest = [int(i) for i in n_hop_neighborhood_of_interest]
    mex_dataframe_HSIC = mex_migration.copy()
    subset_df = mex_dataframe_HSIC[mex_dataframe_HSIC['GEO2_MX'].isin(n_hop_neighborhood_of_interest)]

    subset_df = subset_df.rename({'perc_migrants': 'class'}, axis = 1)
    subset_df = subset_df.reset_index(drop = True)
    all_columns = list(subset_df)
    #put the instance of interest as thee first row to calculate residuals downstream
    row_index = subset_df[subset_df['GEO2_MX'] == instance_of_interest].index[0]
    selected_row = subset_df.iloc[row_index:row_index+1]
    subset_df = subset_df.drop(index=row_index)
    subset_df = pd.concat([selected_row, subset_df], ignore_index=True)
    subset_df = subset_df.reset_index(drop = True)
    export_columns = [i for i in all_columns if i not in ['GEO2_MX', 'Latitude', 'Longitude']]
    subset_df = subset_df[export_columns]
    subset_df.to_csv(f'mex_dataframe_for_HSIC_Lasso_three_hop_municipality_{instance_of_interest}.csv', index = False)



list_of_munis = list(mex_migration['GEO2_MX'])


for item in list_of_munis:
    export_subset_for_HSIC(item)
    


#Once we have the 2-hop neighborhood data exported, we can initiate the HSIC Lasso Explanations


def hsic_lasso_explain_instance(path_to_dataset, number_of_features_k):
    hsic_lasso = HSICLasso()
    hsic_lasso.input(path_to_dataset)
    hsic_lasso.regression(num_feat=number_of_features_k, B=0, M=1)
    explanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient'])
    for i in range(number_of_features_k):
        predictor_name = hsic_lasso.get_features()[i]
        beta_coefficient = hsic_lasso.get_index_score()[i]
        row = {'feature_name' : predictor_name, 'beta_coefficient' : beta_coefficient}
        explanations_df = pd.concat([explanations_df, pd.DataFrame([row])], ignore_index=True)

        #explanations_df = explanations_df.append(row, ignore_index=True)
    
    explanations_df = explanations_df.reindex(explanations_df['beta_coefficient'].abs().sort_values(ascending=False).index)
    
    return explanations_df



#extract top=10 important features for each municipality

#return explanations for each municipality, and export them to wd (to be used to calculate performance metrics)

list_of_issue_ids = []
list_of_zero_division_error_ids = [] #problem with B value in HSIC

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\graph_lime\three_hop")

for item in list_of_munis:
    item = str(item)
    path_to_dataset = fr"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\n_hop_neighbors_subset\three_hop\mex_dataframe_for_HSIC_Lasso_three_hop_municipality_{item}.csv"
    try:
        hsic_explanations_df = hsic_lasso_explain_instance(path_to_dataset, 10)   
        hsic_explanations_df.to_csv(f'HSIC_Lasso_explanation_{item}.csv', index = False)
    except IndexError:
        list_of_issue_ids.append(item)
    except ZeroDivisionError:
        list_of_zero_division_error_ids.append(item)
        
        
list_of_issue_ids.append(list_of_zero_division_error_ids[0])       
        
list_of_issue_ids = [int(i) for i in list_of_issue_ids]



#Extracting preformance measures r-squared and rmse

#For this, as done in the original HSIC-Lasso paper, we will subset trainig data for a given instance of interest to retain only top-k most important variables, and then execute kernel regression

performance_indicators_HSIC_Lasso = pd.DataFrame(columns = ['GEO2_MX', 'adjusted_r_squared', 'mae', 'residual'])

#list_of_munis_HSIC = [i for i in list_of_munis if i not in list_of_issue_ids]
list_of_issue_ids = []

for item in list_of_munis:
    try:
        item = str(item)
        #three hop
        path_to_train_dataset =  fr"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\n_hop_neighbors_subset\three_hop\mex_dataframe_for_HSIC_Lasso_three_hop_municipality_{item}.csv"
        
        #two hop
        #path_to_train_dataset =  f"D:\\vsriva11\\VADER Lab\\migration-portal-VS\\data\\HSIC_Lasso_Datasets\\mex_dataframe_for_HSIC_Lasso_two_hop_municipality_{item}.csv"
        
        hsic_2_hop_sample = pd.read_csv(path_to_train_dataset)
        
        #three hop
        path_to_explanations = fr"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\graph_lime\three_hop\HSIC_Lasso_explanation_{item}.csv"
        
        #two hop
        #path_to_explanations = f"D:\\vsriva11\\VADER Lab\\migration-portal-VS\\data\\Explanations\\HSIC Lasso two hop Explanations\\HSIC_Lasso_explanation_{item}.csv"
    
        hsic_explanations_df = pd.read_csv(path_to_explanations)
        important_features_hsic = hsic_explanations_df['feature_name']
        hsic_2_hop_sample_X = hsic_2_hop_sample[important_features_hsic]
        #since kernel regression is a distance based metric, do standardization
        hsic_2_hop_sample_X = scaler.fit_transform(hsic_2_hop_sample_X)
        hsic_2_hop_sample_Y = hsic_2_hop_sample['class']
        hsic_2_hop_sample_Y = np.array(hsic_2_hop_sample_Y).reshape(-1, 1)    
        hsic_2_hop_sample_Y = scaler.fit_transform(hsic_2_hop_sample_Y)
        hsic_2_hop_sample_Y = np.ravel(hsic_2_hop_sample_Y)    
        
        #fit kernel regression
        regr = SVR(C=1, epsilon=0.2)
        regr.fit(hsic_2_hop_sample_X, hsic_2_hop_sample_Y)
        
        #r squared
        r_squared = regr.score(hsic_2_hop_sample_X, hsic_2_hop_sample_Y)
        #adjusted_rsquared
        n = len(hsic_2_hop_sample_Y)
        p = 10
        if (n - p - 1) != 0:
            adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        else: 
            adjusted_r_squared = 'issue'
        
        #calculate MAE using iriginal scales of the data
        hsic_2_hop_sample_Y_original =  scaler.inverse_transform(hsic_2_hop_sample_Y.reshape(-1, 1))
        predicted_values = regr.predict(hsic_2_hop_sample_X)
        predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1))
        mae = mean_absolute_error(hsic_2_hop_sample_Y_original, predicted_values)
        
        #residual of instance of interest
        residuals = hsic_2_hop_sample_Y_original - predicted_values
        residual_of_interest = residuals[0][0]
    
        
        
        #adjusted_r_squared = r_squared
        row = {'GEO2_MX' : item, 'adjusted_r_squared' : adjusted_r_squared, 'mae' : mae, 'residual' : residual_of_interest}
        #performance_indicators_HSIC_Lasso = performance_indicators_HSIC_Lasso.append(row, ignore_index = True)
        performance_indicators_HSIC_Lasso = pd.concat([performance_indicators_HSIC_Lasso, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError: 
        list_of_issue_ids.append(item)
        continue
    
###Descriptive statistics for local r squared

performance_indicators_HSIC_Lasso = performance_indicators_HSIC_Lasso[performance_indicators_HSIC_Lasso['adjusted_r_squared'] != 'issue']

performance_indicators_HSIC_Lasso = performance_indicators_HSIC_Lasso.reset_index(drop = True)

summary = performance_indicators_HSIC_Lasso['adjusted_r_squared'].describe()
median = performance_indicators_HSIC_Lasso['adjusted_r_squared'].median()
percentiles = np.percentile(performance_indicators_HSIC_Lasso['adjusted_r_squared'], [25, 50, 75], axis=0)
mean = performance_indicators_HSIC_Lasso['adjusted_r_squared'].mean()
std_dev = stdev(performance_indicators_HSIC_Lasso['adjusted_r_squared'])

# Create a summary DataFrame
summary_df_adj_rsquared_hsic = pd.DataFrame({
    'Mean': [mean],
    'Std Dev': [std_dev],
    '25th Percentile': [percentiles[0]],
    'Median': [median],
    '75th Percentile': [percentiles[2]],
    'Max' : max(performance_indicators_HSIC_Lasso['adjusted_r_squared']),
    'Min' : min(performance_indicators_HSIC_Lasso['adjusted_r_squared'])
}, index=['Summary'])



summary_mae = performance_indicators_HSIC_Lasso['mae'].describe()
median_mae = performance_indicators_HSIC_Lasso['mae'].median()
percentiles_mae = np.percentile(performance_indicators_HSIC_Lasso['mae'], [25, 50, 75], axis=0)
mean_mae = performance_indicators_HSIC_Lasso['mae'].mean()
std_dev_mae = stdev(performance_indicators_HSIC_Lasso['mae'])

# Create a summary DataFrame
summary_df_mae = pd.DataFrame({
    'Mean': [mean_mae],
    'Std Dev': [std_dev_mae],
    '25th Percentile': [percentiles_mae[0]],
    'Median': [median_mae],
    '75th Percentile': [percentiles_mae[2]],
    'Max' : max(performance_indicators_HSIC_Lasso['mae']),
    'Min' : min(performance_indicators_HSIC_Lasso['mae'])
}, index=['Summary'])


#For 2-hop:
#failed instances: 13
#Negative r squared: 116
#Negative r squared: 25 


#for 3-hop
#failed instances: 4
#Negative r squared: 3
#Negative r squared: 3





count_negative = 0
count_positive = 0 

for i in range(len(performance_indicators_HSIC_Lasso)):
    if performance_indicators_HSIC_Lasso.loc[i, 'adjusted_r_squared'] <0:
        count_negative+=1 
    elif performance_indicators_HSIC_Lasso.loc[i, 'adjusted_r_squared'] >1:
        count_positive+=1 
        
        
os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations")

summary_df_adj_rsquared_hsic.to_csv('adj_r_squared_performance_metrics_summary_Graph_LIME_three_hop.csv', index = False)
summary_df_mae.to_csv('mae_performance_metrics_summary_Graph_LIME_three_hop.csv', index = False)

performance_indicators_HSIC_Lasso.to_csv("all_performance_indicators_Graph_LIME_three_hop.csv", index = False)


#Moran's I of residuals
#import residuals for GWR LIMe

graph_lime_residuals = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\all_performance_indicators_Graph_LIME_three_hop.csv")

graph_lime_residuals = graph_lime_residuals[['GEO2_MX', 'residual']]

#prepare weights matrix using KNN

#subset mex_migration to keep only available municipalities wrt Graph Lime

mex_migration_graph_lime = pd.merge(mex_migration, graph_lime_residuals, on = 'GEO2_MX', how = 'left')
mex_migration_graph_lime = mex_migration_graph_lime.dropna()




centroid_coordinates = mex_migration_graph_lime[['Longitude', 'Latitude']]

centroid_coordinates = np.array(centroid_coordinates)


w_knn = KNN(centroid_coordinates, k=5)


graph_lime_residuals_array = np.array(mex_migration_graph_lime['residual'])


moran = Moran(graph_lime_residuals_array, w_knn)



print("Moran's I:", moran.I)
print("Expected Moran's I:", moran.EI)
print("p-value:", moran.p_norm)
print("z-score:", moran.z_norm)


####distribution of features####

# list_of_features = []
# failed_instances = []

# for item in list_of_munis:
#     try:
#         explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\graph_lime\two_hop\HSIC_Lasso_explanation_{item}.csv")
#         features = list(explanation['feature_name'])
#         for element in features:
#             list_of_features.append(element)
#     except FileNotFoundError:
#         failed_instances.append(item)
#         continue
        
        

# import matplotlib.pyplot as plt
# from collections import Counter

# category_counts = Counter(list_of_features)

# category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Feature', 'Count'])

# # Extract categories and their frequencies
# sorted_categories = [category for category, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)]
# sorted_frequencies = [category_counts[category] for category in sorted_categories]

# #sorted_frequencies_percentage = [(i/2325)*100 for i in sorted_frequencies]


# plt.bar(sorted_categories, sorted_frequencies, edgecolor='k', width=0.5)
# plt.xticks(rotation='vertical')  # Rotate x-axis labels vertically
# # Add labels and title
# plt.xlabel('Categories')
# plt.ylabel('Frequency')
# plt.title('Top-10 Features Distribution')


# category_counts_df.to_csv('important_feautres_k10_across_municipalites_Graph_LIME.csv', index = False)



#####for the preferred model, get beta weights and top-k most important features for inferential analysis

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime")


list_of_munis = list(mex_migration['GEO2_MX'])


#ols_results_moderate = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, '484014045', 10, return_only_performance_metrics = False)[4]
for item in list_of_munis:
    explanation= OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, str(item), 10, return_only_performance_metrics = False)[4]
    explanation.to_csv(f'OLS_LIME_explanation_{item}.csv', index = False)

    
   

######Prepare a histogram of top-k feautres to see the ones most frequently picked all across globally

list_of_features = []

for item in list_of_munis:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    features = list(explanation['feature_name'])
    for element in features:
        list_of_features.append(element)
        

import matplotlib.pyplot as plt
from collections import Counter

category_counts = Counter(list_of_features)

category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Feature', 'Count'])

category_counts_df['percent_total'] = (category_counts_df['Count']/2325)*100

features_of_interest = ['perc_single_parent_hhtype', 'perc_separated','perc_foreign_born_nativity', 'Satellite_PC_2', 'perc_secondary_edu', 'avg_hrsactual1', 'avg_bedroom_num', 'perc_extended_family_hhtype', 'avg_npersons',  
                        'avg_chsurv', 'avg_chborn', 'perc_married_with_children_hhtype', 'perc_masonry_roof', 'perc_public_administration_defense_indgen',  'perc_other_hlthfac', 'perc_unpaid_worker']
#features based on freq >=50%:
# 0	perc_single_parent_hhtype
# 5	perc_separated
# 6	perc_foreign_born_nativity
# 7	Satellite_PC_2
# 4	perc_secondary_edu
# 11	avg_hrsactual1
# 1	avg_bedroom_num
# 10	perc_extended_family_hhtype
# 3	avg_npersons
# 2	avg_chsurv
# 16	avg_chborn

#features based on novelty:

# 18	perc_married_with_children_hhtype
# 17	perc_masonry_roof
# 9	perc_public_administration_defense_indgen
# 8	perc_other_hlthfac



category_counts_df_subset = category_counts_df[category_counts_df['Feature'].isin(features_of_interest)]

category_counts_df_subset = category_counts_df_subset.sort_values(by='Count', ascending=False)


# Extract categories and their frequencies
# sorted_categories = [category for category, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)]
# sorted_frequencies = [category_counts[category] for category in sorted_categories]

# sorted_frequencies_percentage = [(i/2325)*100 for i in sorted_frequencies]




plt.bar(category_counts_df_subset['Feature'], category_counts_df_subset['percent_total'], edgecolor='k', width=0.5)
plt.xticks(rotation='vertical')  # Rotate x-axis labels vertically
# Add labels and title
plt.xlabel('Features')
plt.ylabel('Percent of Municipalities')
plt.title('Important Features Subset Distribution')

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations")
category_counts_df_subset.to_csv('subset_important_feautres_k10_across_municipalites_OLS_LIME.csv', index = False)

#For mapping, import all explanations across municipalities, make a list of unique features, and then for each feature, record GEO2_MX and beta coefficients

all_explanations_df = pd.DataFrame(columns = {'GEO2_MX', 'feature_name', 'beta_coefficient'})


for item in list_of_munis:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['GEO2_MX'] = item
    all_explanations_df = pd.concat([all_explanations_df, explanation], axis=0)


#unique features

unique_features = list(set(all_explanations_df['feature_name']))


os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\map_data\ols_lime")

for item in unique_features:
    #unique_feature_df = pd.DataFrame(columns = ['GEO2_MX', item])
    feature_subset = all_explanations_df[all_explanations_df['feature_name'] == item]
    feature_subset = feature_subset[['GEO2_MX', 'beta_coefficient']]
    feature_subset = feature_subset.rename({'beta_coefficient' : item}, axis = 1)
    feature_subset.to_csv(f'beta_coefficients_{item}.csv', index = False)
    
    




###Experiment: fit a global ols model to determine if the fit improves, and if Moran's I of residuals improve as well for local models

mex_migration_global = mex_migration.copy()


mex_migration_global['GEO2_MX'] = mex_migration_global['GEO2_MX'].astype(str)

all_columns = list(mex_migration_global.columns)
predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', 'perc_migrants', 'Latitude', 'Longitude']]


predictor_dataframe = mex_migration_global[predictor_columns]

#make a copy to use for correlation analysis later in the pipeline
predictor_dataframe_for_correlation = predictor_dataframe.copy()


predictor_dataframe = scaler.fit_transform(predictor_dataframe)

X = predictor_dataframe.copy()

predicted_dataframe = mex_migration_global[['perc_migrants']] 

y = predicted_dataframe.values.reshape((-1,1)) # reshape is needed to have column array


#run lasso globally across the entire dataset, fetch top 10 most important features



alphas, _, coefs = lars_path(X,
                             np.ravel(y),
                             method='lasso',
                             verbose=False)

#number_of_features_k = 10
for i in range(len(coefs.T) - 1, 0, -1):
    nonzero = coefs.T[i].nonzero()[0]
    if len(nonzero) <= 10:
        break

important_features_global = nonzero

model_regressor = LinearRegression(fit_intercept=True)

easy_model = model_regressor

easy_model.fit(X[:, important_features_global], y)



global_r_squared = easy_model.score( X[:, important_features_global], y)
n = len(y)
p = 10
global_adjusted_r_squared = 1 - ((1 - global_r_squared) * (n - 1) / (n - p - 1))


#calculate global MAE
predicted_value_global = easy_model.predict(X[:, important_features_global])


global_residuals = y - predicted_value_global

global_mae = np.mean(np.abs(global_residuals))

#Check Moran's I of residuals
centroid_coordinates = mex_migration_global[['Longitude', 'Latitude']]

centroid_coordinates = np.array(centroid_coordinates)


w_knn = KNN(centroid_coordinates, k=5)


#graph_lime_residuals_array = np.array(mex_migration_graph_lime['residual'])


moran = Moran(np.array(global_residuals), w_knn)



print("Moran's I:", moran.I)
print("Expected Moran's I:", moran.EI)
print("p-value:", moran.p_norm)
print("z-score:", moran.z_norm)


##########Experiment 2: Stability wrt k parameter############

#randomly select 20 instances
list_of_munis = list(mex_migration['GEO2_MX'])

# Number of instances to randomly select
num_instances_to_select = 465 #(20% of the dataset)


selected_instances = random.sample(list_of_munis, num_instances_to_select)


def return_beta_weight_tracker(instance_of_interest, method): #instance_of_interest given in int, method: GWR, OLS, HSIC
    instance_of_interest = str(instance_of_interest)

    beta_weights_tracker_df = pd.DataFrame(columns = ['feature', 'beta_wt_path'])
    problem_name = []
    problem_k = []
     #tracking from k=5 to k=20, for a given instance
    for i in range(5, 21):
        features_already_recorded = list(beta_weights_tracker_df['feature'])
        if method == 'GWR':
            instance_explanation = gwr_lasso_explain_instance(instance_of_interest, i, return_only_performance_metrics = False)[0]
        elif method == 'OLS':
            instance_explanation = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, instance_of_interest, i, return_only_performance_metrics = False)[4]
        elif method == 'HSIC':
            try:
                path_to_dataset = rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\n_hop_neighbors_subset\three_hop\mex_dataframe_for_HSIC_Lasso_three_hop_municipality_{instance_of_interest}.csv"
                instance_explanation = hsic_lasso_explain_instance(path_to_dataset, i)
            except IndexError:
                #print(instance_of_interest, ' did not return values at ', i)
                problem_name.append(instance_of_interest)
                problem_k.append(i)
                continue
            except FileNotFoundError:
                break

        instance_explanation = instance_explanation.reset_index(drop = True)
        
        #print(list(instance_explanation['feature_name']))
        
        # if 'total_pop' not in list(instance_explanation['feature_name']):
        #     print('total_pop not in k= ', i )
        
        for j in range(len(instance_explanation)):
            if instance_explanation.loc[j, 'feature_name'] not in features_already_recorded:
                row = {'feature': instance_explanation.loc[j, 'feature_name'], 'beta_wt_path' : str(instance_explanation.loc[j, 'beta_coefficient'])}
                beta_weights_tracker_df = pd.concat([beta_weights_tracker_df, pd.DataFrame([row])], ignore_index=True)
    
            else:
                condition = beta_weights_tracker_df['feature'] == instance_explanation.loc[j, 'feature_name'] 
                selected_index = beta_weights_tracker_df.index[condition].tolist()[0]
                #selected_row = beta_weights_tracker_df[condition]
    
                beta_weights_tracker_df.loc[selected_index, 'beta_wt_path'] = beta_weights_tracker_df.loc[selected_index, 'beta_wt_path'] + ', ' + str(instance_explanation.loc[j, 'beta_coefficient'])
                
                
    beta_weights_tracker_df['beta_wt_path'] =  beta_weights_tracker_df['beta_wt_path'].apply(lambda x: x.split(', '))      
            
            
    def string_list_to_floats(string_list):
        return [float(num) for num in string_list]
    
    # Apply the custom function to the specified column
    beta_weights_tracker_df['beta_wt_path'] = beta_weights_tracker_df['beta_wt_path'].apply(string_list_to_floats)
        
    beta_weights_tracker_df['z_scores'] =  beta_weights_tracker_df['beta_wt_path'].apply(lambda x: stats.zscore(x)) 
    #beta_weights_tracker_df['std_dev'] =  beta_weights_tracker_df['beta_wt_path'].apply(lambda x: np.std(np.array(x)))  
    beta_weights_tracker_df['range_z_scores'] =  beta_weights_tracker_df['z_scores'].apply(lambda x: np.ptp(x))
    
    
    beta_weights_tracker_df = beta_weights_tracker_df.fillna(0)
    mean_range_z_scores = np.mean(list(beta_weights_tracker_df['range_z_scores']))
    
    return [beta_weights_tracker_df, mean_range_z_scores, problem_name, problem_k]



##Stability metrics 

mean_range_z_scores_random_samples = []

for item in selected_instances:
    mean_range_z = return_beta_weight_tracker(item, method = 'GWR')[1]
    mean_range_z_scores_random_samples.append(mean_range_z)
    
    
final_range_z_scores_across_samples = np.mean(mean_range_z_scores_random_samples)

mean_range_z_scores_random_samples_df = pd.DataFrame(mean_range_z_scores_random_samples, columns = ['mean_range_z_scores'])

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants")
mean_range_z_scores_random_samples_df.to_csv('mean_range_z_scores_random_samples_df_GWR_LIME.csv', index = False)



##Stability metrics for OLS LIME

mean_range_z_scores_random_samples_OLS = []
for item in selected_instances:
    mean_range_z = return_beta_weight_tracker(item, method = 'OLS')[1]
    mean_range_z_scores_random_samples_OLS.append(mean_range_z)
    
   
########to execute 10/24
final_range_z_scores_across_samples_OLS = np.mean(mean_range_z_scores_random_samples_OLS)

mean_range_z_scores_random_samples_OLS_df = pd.DataFrame(mean_range_z_scores_random_samples_OLS, columns = ['mean_range_z_scores'])

mean_range_z_scores_random_samples_OLS_df.to_csv('mean_range_z_scores_random_samples_df_OLS_LIME.csv', index = False)

#######to execute 10/24

##Stability metrics for Graph LIME

mean_range_z_scores_random_samples_HSIC = []
names = []
ks = []

for item in selected_instances:
    mean_range_z = return_beta_weight_tracker(item, method = 'HSIC')[1]
    problem_names = return_beta_weight_tracker(item, method = 'HSIC')[2]
    problem_ks = return_beta_weight_tracker(item, method = 'HSIC')[3]
    mean_range_z_scores_random_samples_HSIC.append(mean_range_z)
    names.append(problem_names)
    ks.append(problem_ks)

import math
mean_range_z_scores_random_samples_HSIC = [x for x in mean_range_z_scores_random_samples_HSIC if not math.isnan(x)]

    
final_range_z_scores_across_samples_HSIC = np.mean(mean_range_z_scores_random_samples_HSIC)


mean_range_z_scores_random_samples_HSIC_df = pd.DataFrame(mean_range_z_scores_random_samples_HSIC, columns = ['mean_range_z_scores'])


mean_range_z_scores_random_samples_HSIC_df.to_csv('mean_range_z_scores_random_samples_df_GRAPH_LIME.csv', index = False)


####Experiment/Analysis: Check the overlap of geo-wts vs feature weights###############

import scipy.stats as stats
list_of_munis = list(mex_migration['GEO2_MX'])


geo_wts = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\neighborhood_weights\GWR-LIME\GWR_Weights_fixed_gaussian_bandwidth_2023_09_29_10_17.csv")

kendall_tau_df = pd.DataFrame(columns = ['GEO2_MX', 'kendalls_tau', 'p_value'])




for item in list_of_munis:
    #import feature wts (for a sample)

    feature_wts_sample = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\neighborhood_weights\OLS-LIME\OLS_LIME_weights_{item}.csv")
    
    #sort descending
    
    feature_wts_sample = feature_wts_sample.sort_values(by='feature weights', ascending=False)
    
    ordered_list_feature_wts = list(feature_wts_sample['GEO2_MX'])
    #ordered_list_feature_wts = np.array(ordered_list_feature_wts)
    
    #import geo-wts
    
    geo_wts_sample = geo_wts[geo_wts['GEO2_MX'] == item]
    
    geo_wts_sample['weights'] = geo_wts_sample['weights'].apply(ast.literal_eval)
    
    geo_wts_sample = geo_wts_sample.explode('weights').reset_index(drop = True)
    
    geo_wts_sample = pd.merge(geo_wts_sample[['weights']], geo_wts[['GEO2_MX']], left_index = True, right_index = True, how = 'left')
    
    geo_wts_sample = geo_wts_sample.sort_values(by='weights', ascending=False)
    
    ordered_list_geo_wts = list(geo_wts_sample['GEO2_MX'])
    #ordered_list_geo_wts = np.array(ordered_list_geo_wts)
    

    
    tau, p_value = stats.kendalltau(ordered_list_feature_wts, ordered_list_geo_wts)
    row = {'GEO2_MX' : item, 'kendalls_tau' : tau , 'p_value' : p_value}
    kendall_tau_df = pd.concat([kendall_tau_df, pd.DataFrame([row])], ignore_index=True)


#count number of cases where feature space and geop space are similar


similar_instances = []

for i in range(len(kendall_tau_df)):
    if (kendall_tau_df.loc[i, 'p_value'] <=0.05) and (kendall_tau_df.loc[i, 'kendalls_tau'] >=0):
        similar_instances.append(kendall_tau_df.loc[i, 'GEO2_MX'])
        
        
#########Experiment/Analysis: Determine clusters using local Moran's I wrt percent migrants

centroid_coordinates = mex_migration[['Longitude', 'Latitude']]

centroid_coordinates = np.array(centroid_coordinates)


w_knn = KNN(centroid_coordinates, k=5)


moran_local = Moran_Local(mex_migration['perc_migrants'].values, w_knn)


moran_local_per_muni = moran_local.Is

moran_local_cluster_type_per_muni = moran_local.q

p_values = moran_local.p_sim

local_moran_stats = pd.DataFrame(columns = ['cluster_type', 'p_value'])

for i, p_value in enumerate(p_values):
    cluster_type = moran_local.q[i]
    row = {'cluster_type': cluster_type, 'p_value': p_value}
    local_moran_stats = pd.concat([local_moran_stats, pd.DataFrame([row])], ignore_index = True)
    #local_moran_stats.append(row, ignore_index=True)
    

no_of_significant_MoranI = 0
cluster_type_of_significant_moran = []

for i in range(len(local_moran_stats)):
    if local_moran_stats.loc[i, 'p_value'] <=0.05:
        no_of_significant_MoranI+=1
        cluster_type_of_significant_moran.append(local_moran_stats.loc[i, 'cluster_type'])

##Prepare Local-Moran for mapping (1 HH, 2 LH, 3 LL, 4 HL)


#GEO2_MX, Cluster_type, p-value, then fuilter the rows based on siognificanl p-value (<=0.05)

local_moran_stats = pd.merge(local_moran_stats, mex_migration[['GEO2_MX']], left_index = True, right_index = True, how = 'left')


local_moran_stats = local_moran_stats[local_moran_stats['p_value'] <= 0.05]

#change integers to classes

#Define a dictionary to specify the replacements
replacement_dict = {1: 'H-H', 2: 'L-H', 3: 'L-L', 4: 'H-L'}

#Use the replace() method to replace values in the column
local_moran_stats['cluster_type'] = local_moran_stats['cluster_type'].replace(replacement_dict)


local_moran_stats.to_csv('predicted_perc_migrants_local_moran_I_clusters.csv', index = False)


##Analyzing differeces between H-H and L-L cluster instances

local_moran_stats = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\predicted_perc_migrants_local_moran_I_clusters.csv")

high_high_clusters = local_moran_stats[local_moran_stats['cluster_type'] == 'H-H']


high_high_clusters_ids = list(high_high_clusters['GEO2_MX'])

low_low_clusters = local_moran_stats[local_moran_stats['cluster_type'] == 'L-L']


low_low_clusters_ids = list(low_low_clusters['GEO2_MX'])

#analysis 1: Any major diff between the most frequently selected features (Use Jaccard Similarity)


list_of_features_high_high = []

for item in high_high_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    features = list(explanation['feature_name'])
    for element in features:
        list_of_features_high_high.append(element)
        

list_of_features_high_high_jaccard = list(set(list_of_features_high_high))


list_of_features_low_low = []

for item in low_low_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    features = list(explanation['feature_name'])
    for element in features:
        list_of_features_low_low.append(element)
        

list_of_features_low_low_jaccard = list(set(list_of_features_low_low))

#Jaccard similarity

intersection = len(set(list_of_features_high_high_jaccard).intersection(list_of_features_low_low_jaccard))
union = len(set(list_of_features_high_high_jaccard).union(list_of_features_low_low_jaccard))
jaccard_similarity = intersection / union

print("Jaccard Similarity:", jaccard_similarity)
#Jaccard similarity is 0.38, mainly becuase there are more unique features in low-low cluster


#analysis 2: Any major diff between the order/ranking of the selected features (Use Kendall Coefficient)

high_high_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in high_high_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    high_high_exlanations_df = pd.concat([high_high_exlanations_df, explanation], axis = 0)


rank_dict_high_high = {}

for i in range(1, 11):
    rank_subset = high_high_exlanations_df[high_high_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    category_counts['percent'] = (category_counts['counts']/len(high_high_clusters_ids))*100
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_high_high[i] = top_feature

    
#ties for ranks 2, 3 (perc_extended_family_hhtype), 5,6 (Satellite_PC_2), 8, 9 (avg_hrsactual1)


low_low_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in low_low_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    low_low_exlanations_df = pd.concat([low_low_exlanations_df, explanation], axis = 0)


rank_dict_low_low = {}

for i in range(1, 11):
    rank_subset = low_low_exlanations_df[low_low_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    category_counts['percent'] = (category_counts['counts']/len(low_low_clusters_ids))*100
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_low_low[i] = top_feature


#kendall tau



tau= np.correlate(list(rank_dict_high_high.keys()), list(rank_dict_low_low.keys()))

tau, p_value = stats.kendalltau(list(rank_dict_high_high.values()), list(rank_dict_low_low.values()))



#H-L cluster, L-H cluster

high_low_clusters = local_moran_stats[local_moran_stats['cluster_type'] == 'H-L']


high_low_clusters_ids = list(high_low_clusters['GEO2_MX'])
    

low_high_clusters = local_moran_stats[local_moran_stats['cluster_type'] == 'L-H']


low_high_clusters_ids = list(low_high_clusters['GEO2_MX'])




high_low_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in high_low_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    high_low_exlanations_df = pd.concat([high_low_exlanations_df, explanation], axis = 0)


rank_dict_high_low = {}

for i in range(1, 11):
    rank_subset = high_low_exlanations_df[high_low_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_high_low[i] = top_feature




low_high_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in low_high_clusters_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    low_high_exlanations_df = pd.concat([low_high_exlanations_df, explanation], axis = 0)


rank_dict_low_high = {}

for i in range(1, 11):
    rank_subset = low_high_exlanations_df[low_high_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_low_high[i] = top_feature



#globally, an analysis of rankings
all_explanations = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

all_ids = list(mex_migration['GEO2_MX'])


for item in all_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    all_explanations = pd.concat([all_explanations, explanation], axis = 0)


rank_dict_global = {}

for i in range(1, 11):
    rank_subset = all_explanations[all_explanations['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    category_counts['percent'] = (category_counts['counts']/len(all_ids))*100
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_global[i] = top_feature


#extreme cases (municipalities each with highest and lowest predicted migrations in percentiles, top and bptoom 5 percentiles basically)

median = mex_migration['perc_migrants'].median()
percentiles = np.percentile(mex_migration['perc_migrants'], [5, 10, 25, 50, 75, 95], axis=0)
mean = mex_migration['perc_migrants'].mean()
std_dev = stdev(mex_migration['perc_migrants'])

# Create a summary DataFrame
summary_df_perc_migrants = pd.DataFrame({
    'Mean': [mean],
    'Std Dev': [std_dev],
    '5th Percentile': [percentiles[0]],
    'Median': [median],
    '95th Percentile': [percentiles[4]],
    
    'Max' : max(mex_migration['perc_migrants']),
    'Min' : min(mex_migration['perc_migrants'])
}, index=['Summary'])


five_percentile_ids = mex_migration[mex_migration['perc_migrants']==0]

five_percentile_ids = list(five_percentile_ids['GEO2_MX'])

ninty_five_percentile_ids = mex_migration[mex_migration['perc_migrants']>=8]

ninty_five_percentile_ids = list(ninty_five_percentile_ids['GEO2_MX'])

five_percentile_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in five_percentile_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    five_percentile_exlanations_df = pd.concat([five_percentile_exlanations_df, explanation], axis = 0)


rank_dict_five_percentile = {}

for i in range(1, 11):
    rank_subset = five_percentile_exlanations_df[five_percentile_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    category_counts['percent'] = (category_counts['counts']/len(five_percentile_ids))*100
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_five_percentile[i] = top_feature



ninty_five_percentile_exlanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'rank'])

for item in ninty_five_percentile_ids:
    explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
    explanation['rank'] = (explanation.index + 1)
    ninty_five_percentile_exlanations_df = pd.concat([ninty_five_percentile_exlanations_df, explanation], axis = 0)


rank_dict_ninty_five_percentile = {}

for i in range(1, 11):
    rank_subset = ninty_five_percentile_exlanations_df[ninty_five_percentile_exlanations_df['rank'] == i]
    category_counts = pd.DataFrame(rank_subset['feature_name'].value_counts()).rename(columns = {'feature_name': 'counts'})
    category_counts['feature_name'] = category_counts.index
    category_counts = category_counts.reset_index(drop = True)
    category_counts['percent'] = (category_counts['counts']/len(ninty_five_percentile_ids))*100
    top_feature = category_counts.loc[0, 'feature_name']
    rank_dict_ninty_five_percentile[i] = top_feature




##For a group of features, whck for extrme cases, if there are significant differences using independent t test
from scipy import stats
features_to_compare = ['perc_single_parent_hhtype', 'perc_extended_family_hhtype', 'perc_secondary_edu', 'Satellite_PC_2', 'avg_bedroom_num', 'avg_hrsactual1']

five_percentile_subset = mex_migration[mex_migration['GEO2_MX'].isin(five_percentile_ids)]
ninty_five_percentile_subset = mex_migration[mex_migration['GEO2_MX'].isin(ninty_five_percentile_ids)]

five_percentile_subset = five_percentile_subset[features_to_compare]
ninty_five_percentile_subset = ninty_five_percentile_subset[features_to_compare]

H_H_subset = mex_migration[mex_migration['GEO2_MX'].isin(high_high_clusters_ids)]
L_L_subset = mex_migration[mex_migration['GEO2_MX'].isin(low_low_clusters_ids)]

H_H_subset = H_H_subset[features_to_compare]
L_L_subset = L_L_subset[features_to_compare]



perc_single_parent_hhtype_five = np.array(L_L_subset['Satellite_PC_2'])
perc_single_parent_hhtype_ninty_five = np.array(H_H_subset['Satellite_PC_2'])


perc_single_parent_hhtype_five_mean = perc_single_parent_hhtype_five.mean()
perc_single_parent_hhtype_ninty_five_mean = perc_single_parent_hhtype_ninty_five.mean()

t_statistic, p_value = stats.ttest_ind(perc_single_parent_hhtype_five, perc_single_parent_hhtype_ninty_five, equal_var=True)


###Frequency distribution for features at each rank, for H-H, L-L, 5th percentile, 95th percentile

high_high_clusters_ids, low_low_clusters_ids




def extract_ranked_feature(group_type, rank):
    cluster_wise_feature_rank = pd.DataFrame(columns = ['GEO2_MX', 'feature_name'])
    if group_type == 'HH':
        ids = high_high_clusters_ids.copy()
    elif group_type == 'LL':
        ids = low_low_clusters_ids.copy()
    elif group_type == '5_percentile':
        ids = five_percentile_ids.copy()   
    elif group_type == '95_percentile':
        ids = ninty_five_percentile_ids.copy()   

    for item in ids:
        try:
            explanation = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\ols_lime\OLS_LIME_explanation_{item}.csv")
            explanation['rank'] = (explanation.index + 1)
            explanation = explanation[explanation['rank'] == rank]
            explanation = explanation.reset_index(drop = True)
            row = {'GEO2_MX' : item, 'feature_name': explanation.loc[0, 'feature_name'] }
            cluster_wise_feature_rank = pd.concat([cluster_wise_feature_rank, pd.DataFrame([row])], ignore_index=True)
        except KeyError:
            print(item)
            continue
            
    return cluster_wise_feature_rank
            
        
        
H_H_rank_1 = extract_ranked_feature('HH', 1)
H_H_rank_2 = extract_ranked_feature('HH', 2)
H_H_rank_3 = extract_ranked_feature('HH', 3)
H_H_rank_4 = extract_ranked_feature('HH', 4)
H_H_rank_5 = extract_ranked_feature('HH', 5)
H_H_rank_6 = extract_ranked_feature('HH', 6)
H_H_rank_7 = extract_ranked_feature('HH', 7)
H_H_rank_8 = extract_ranked_feature('HH', 8)
H_H_rank_9 = extract_ranked_feature('HH', 9)
H_H_rank_10 = extract_ranked_feature('HH', 10)



L_L_rank_1 = extract_ranked_feature('LL', 1)
L_L_rank_2 = extract_ranked_feature('LL', 2)
L_L_rank_3 = extract_ranked_feature('LL', 3)
L_L_rank_4 = extract_ranked_feature('LL', 4)
L_L_rank_5 = extract_ranked_feature('LL', 5)
L_L_rank_6 = extract_ranked_feature('LL', 6)
L_L_rank_7 = extract_ranked_feature('LL', 7)
L_L_rank_8 = extract_ranked_feature('LL', 8)
L_L_rank_9 = extract_ranked_feature('LL', 9)
L_L_rank_10 = extract_ranked_feature('LL', 10)

five_percentile_rank_1 = extract_ranked_feature('5_percentile', 1)
five_percentile_rank_2 = extract_ranked_feature('5_percentile', 2)
five_percentile_rank_3 = extract_ranked_feature('5_percentile', 3)
five_percentile_rank_4 = extract_ranked_feature('5_percentile', 4)
five_percentile_rank_5 = extract_ranked_feature('5_percentile', 5)
five_percentile_rank_6 = extract_ranked_feature('5_percentile', 6)
five_percentile_rank_7 = extract_ranked_feature('5_percentile', 7)
five_percentile_rank_8 = extract_ranked_feature('5_percentile', 8)
five_percentile_rank_9 = extract_ranked_feature('5_percentile', 9)
five_percentile_rank_10 = extract_ranked_feature('5_percentile', 10)

ninty_five_percentile_rank_1 = extract_ranked_feature('95_percentile', 1)
ninty_five_percentile_rank_2 = extract_ranked_feature('95_percentile', 2)
ninty_five_percentile_rank_3 = extract_ranked_feature('95_percentile', 3)
ninty_five_percentile_rank_4 = extract_ranked_feature('95_percentile', 4)
ninty_five_percentile_rank_5 = extract_ranked_feature('95_percentile', 5)
ninty_five_percentile_rank_6 = extract_ranked_feature('95_percentile', 6)
ninty_five_percentile_rank_7 = extract_ranked_feature('95_percentile', 7)
ninty_five_percentile_rank_8 = extract_ranked_feature('95_percentile', 8)
ninty_five_percentile_rank_9 = extract_ranked_feature('95_percentile', 9)
ninty_five_percentile_rank_10 = extract_ranked_feature('95_percentile', 10)


#A final table that has geo2_MX, group_type, and perc_migrants columns for visualization

H_H_subset = mex_migration[mex_migration['GEO2_MX'].isin(high_high_clusters_ids)]

H_H_subset = H_H_subset[['GEO2_MX', 'perc_migrants']]
H_H_subset['cluster'] = 'H-H'


L_L_subset = mex_migration[mex_migration['GEO2_MX'].isin(low_low_clusters_ids)]

L_L_subset = L_L_subset[['GEO2_MX', 'perc_migrants']]
L_L_subset['cluster'] = 'L-L'

L_H_subset = mex_migration[mex_migration['GEO2_MX'].isin(low_high_clusters_ids)]

L_H_subset = L_H_subset[['GEO2_MX', 'perc_migrants']]
L_H_subset['cluster'] = 'L-H'

H_L_subset = mex_migration[mex_migration['GEO2_MX'].isin(high_low_clusters_ids)]

H_L_subset = H_L_subset[['GEO2_MX', 'perc_migrants']]
H_L_subset['cluster'] = 'H-L'

five_percentile_subset = mex_migration[mex_migration['GEO2_MX'].isin(five_percentile_ids)]

five_percentile_subset = five_percentile_subset[['GEO2_MX', 'perc_migrants']]
five_percentile_subset['percentile_group'] = 'five_percentile'

ninty_five_percentile_subset = mex_migration[mex_migration['GEO2_MX'].isin(ninty_five_percentile_ids)]

ninty_five_percentile_subset = ninty_five_percentile_subset[['GEO2_MX', 'perc_migrants']]
ninty_five_percentile_subset['percentile_group'] = 'ninty_five_percentile'




cluster_subset = pd.concat([H_H_subset, L_L_subset, L_H_subset, H_L_subset], axis = 0)


os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\Groups\cluster_based")

cluster_subset.to_csv('instances_grouped_by_cluster_type.csv', index = False)

H_H_rank_1.to_csv('H_H_rank_1.csv', index = False)
H_H_rank_2.to_csv('H_H_rank_2.csv', index = False)
H_H_rank_3.to_csv('H_H_rank_3.csv', index = False) 
H_H_rank_4.to_csv('H_H_rank_4.csv', index = False)
H_H_rank_5.to_csv('H_H_rank_5.csv', index = False) 
H_H_rank_6.to_csv('H_H_rank_6.csv', index = False)
H_H_rank_7.to_csv('H_H_rank_7.csv', index = False) 
H_H_rank_8.to_csv('H_H_rank_8.csv', index = False)
H_H_rank_9.to_csv('H_H_rank_9.csv', index = False)
H_H_rank_10.to_csv('H_H_rank_10.csv', index = False)

L_L_rank_1.to_csv('L_L_rank_1.csv', index = False)
L_L_rank_2.to_csv('L_L_rank_2.csv', index = False)
L_L_rank_3.to_csv('L_L_rank_3.csv', index = False) 
L_L_rank_4.to_csv('L_L_rank_4.csv', index = False)
L_L_rank_5.to_csv('L_L_rank_5.csv', index = False) 
L_L_rank_6.to_csv('L_L_rank_6.csv', index = False)
L_L_rank_7.to_csv('L_L_rank_7.csv', index = False) 
L_L_rank_8.to_csv('L_L_rank_8.csv', index = False)
L_L_rank_9.to_csv('L_L_rank_9.csv', index = False)
L_L_rank_10.to_csv('L_L_rank_10.csv', index = False)








percentile_subset = pd.concat([five_percentile_subset, ninty_five_percentile_subset], axis = 0)


os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\Groups\percentile_based")

percentile_subset.to_csv('instances_grouped_by_percentile.csv', index = False)

five_percentile_rank_1.to_csv('five_percentile_rank_1.csv', index = False)
five_percentile_rank_2.to_csv('five_percentile_rank_2.csv', index = False)
five_percentile_rank_3.to_csv('five_percentile_rank_3.csv', index = False) 
five_percentile_rank_4.to_csv('five_percentile_rank_4.csv', index = False)
five_percentile_rank_5.to_csv('five_percentile_rank_5.csv', index = False) 
five_percentile_rank_6.to_csv('five_percentile_rank_6.csv', index = False)
five_percentile_rank_7.to_csv('five_percentile_rank_7.csv', index = False) 
five_percentile_rank_8.to_csv('five_percentile_rank_8.csv', index = False)
five_percentile_rank_9.to_csv('five_percentile_rank_9.csv', index = False)
five_percentile_rank_10.to_csv('five_percentile_rank_10.csv', index = False)

ninty_five_percentile_rank_1.to_csv('ninty_five_percentile_rank_1.csv', index = False)
ninty_five_percentile_rank_2.to_csv('ninty_five_percentile_rank_2.csv', index = False)
ninty_five_percentile_rank_3.to_csv('ninty_five_percentile_rank_3.csv', index = False) 
ninty_five_percentile_rank_4.to_csv('ninty_five_percentile_rank_4.csv', index = False)
ninty_five_percentile_rank_5.to_csv('ninty_five_percentile_rank_5.csv', index = False) 
ninty_five_percentile_rank_6.to_csv('ninty_five_percentile_rank_6.csv', index = False)
ninty_five_percentile_rank_7.to_csv('ninty_five_percentile_rank_7.csv', index = False) 
ninty_five_percentile_rank_8.to_csv('ninty_five_percentile_rank_8.csv', index = False)
ninty_five_percentile_rank_9.to_csv('ninty_five_percentile_rank_9.csv', index = False)
ninty_five_percentile_rank_10.to_csv('ninty_five_percentile_rank_10.csv', index = False)



#For H-H group, send beta values for avg_ch_boorn
chborn_df = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\map_data\ols_lime\beta_coefficients_avg_chborn.csv")

HH_chborn_df = chborn_df[chborn_df['GEO2_MX'].isin(high_high_clusters_ids)]

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\map_data\ols_lime")
HH_chborn_df.to_csv('H_H_avg_chborn.csv', index = False)

ninty_five_percentile_chborn_df = chborn_df[chborn_df['GEO2_MX'].isin(ninty_five_percentile_ids)]

ninty_five_percentile_chborn_df.to_csv('ninety_five_percentile_avg_chborn.csv', index = False)


married_w_children = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\top_k_important_features\k_10\map_data\ols_lime\beta_coefficients_perc_married_with_children_hhtype.csv")
 
five_percentile_married_w_children = married_w_children[married_w_children['GEO2_MX'].isin(five_percentile_ids)]


five_percentile_married_w_children.to_csv('five_percentile_married_w_children.csv', index = False)

   #rank_dict_global[i] = top_feature



# import matplotlib.pyplot as plt
# from collections import Counter

# category_counts = Counter(list_of_features)

# category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Feature', 'Count'])


####Send list of ID's to shade in black (they were never a part of the analysis)

original_shape_file =  gpd.read_file(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\ipumns_shp.shp")

original_list = list(original_shape_file['shapeID'])

original_list = [int(i) for i in original_list ]

analyzed_list = list(mex_migration['GEO2_MX'])


excluded_municipalities = [i for i in original_list if i not in analyzed_list]


# Specify the path and filename for the JSON file

os.chdir(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff")
json_file_path = "municipalities_excluded_from_analysis.json"

# Export the list as a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(excluded_municipalities, json_file)

print(f"The list has been exported to {json_file_path}")



###analyzing variable groupings




variable_grouping_df = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\mexico_migration_with_crime_no_implicit_variable_grouping.csv")


group_counts = variable_grouping_df.groupby('feature_group').size().reset_index(name='Count')

color_palette = {'Economic': '#1f77b4', 'Health': '#ff7f0e', 'Accessibility': '#2ca02c', 'Environment': '#d62728', 'Housing': '#9467bd', 'Household unit': '#8c564b', 'Satellite': '#e377c2', 'Demographic': '#17becf', 'Education': '#7f7f7f', 'Safety': '#bcbd22'}



####Creating bar graphs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np




def feature_importance_plotter(municipality_index, explanation_method, legend_position = 1.34, include_satellite = True):
    #aguascalientes
    explanations = pd.read_csv(rf"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\Explanations\target total_migrants\top_k_important_features\Original_top_10_features\{explanation_method}_LIME\{explanation_method}_LIME_explanation_{municipality_index}.csv")
    
    if include_satellite == False:
        explanations = explanations[~explanations['feature_name'].str.contains('Satellite')]
        explanations = explanations.reset_index(drop = True)
        
    
    #append feature groupings
    explanations = pd.merge(explanations, variable_grouping_df, on = 'feature_name', how = 'left')
    explanations['abs_beta'] = explanations['beta_coefficient'].abs()
    
    explanations = explanations.sort_values(by='abs_beta', ascending = True)
    
    #convert beta coefficients to log values for easier visualization
    beta_coefficients = np.array(explanations['beta_coefficient'])
    log_beta_coefficients = np.sign(beta_coefficients) * np.log10(np.abs(beta_coefficients))
    
    
    
    
    features = list(explanations['feature_name'])
    values = list(log_beta_coefficients)
    categories = list(explanations['feature_group'])
    colors = {}
    
    for i in range(len(categories)):
        colors[categories[i]] = color_palette[categories[i]]
        
    
    
    fig, ax = plt.subplots()
    
    # Plot horizontal bar chart
    #bars = ax.barh(features, values, color=['green' if v >= 0 else 'red' for v in values])
    bars = ax.barh(features, values, color=explanations['feature_group'].map(colors))
    
    # Add y-axis origin line
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Add labels and title
    ax.set_xlabel('log beta coefficient')
    if explanation_method == 'GWR':
        ax.set_title('GW-LIME Explanations')
    else:
        ax.set_title('LIME Explanations')
        
    
    # Add data values next to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width >= 0 else width  # Adjust the offset for negative values
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, str(round(width, 2)),
                va='center', ha='left' if width >= 0 else 'right', color='black')
    # Remove bounding box around the plot
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # Adjusting the left and right bounds
    #plt.xlim(left=-300, right=300)  # Set the desired bounds
    
    ax.set_xlim(left=min(values) - 1, right=max(values) + 1)  # Set the desired bounds
    
    # Creating legend handles
    legend_handles = [Patch(color=colors[c], label=c) for c in list(set(categories))]
    
    # Adding legend
    ax.legend(handles=legend_handles, loc='lower right', borderpad=0.0011, bbox_to_anchor=(legend_position, 0))
    
    
    #plt.tight_layout()
    # Show the plot
    plt.show()


#GWR plots
feature_importance_plotter('484001002', 'GWR', legend_position = 1.34) #aguascalientes

feature_importance_plotter('484012051', 'GWR', legend_position = 1.27) #guerrero


feature_importance_plotter('484015100', 'GWR', legend_position = 1.32) #state of mexico

feature_importance_plotter('484016057', 'GWR', legend_position = 1.30) #Michoacán


#OLS plots

feature_importance_plotter('484001002', 'OLS', legend_position = 1.34) #aguascalientes

feature_importance_plotter('484012051', 'OLS', legend_position = 1.34) #guerrero


feature_importance_plotter('484015100', 'OLS') #state of mexico

feature_importance_plotter('484016057', 'OLS') #Michoacán

























