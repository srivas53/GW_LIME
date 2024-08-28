# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:46:54 2023

@author: vsriva11
"""
#Import packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from libpysal  import weights
from libpysal.weights import Queen, KNN
import libpysal.weights as wts
from esda.moran import Moran, Moran_Local
from giddy.directional import Rose
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from statistics import stdev
import ast
from sklearn.linear_model import Ridge, lars_path, LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import statsmodels.api as sm
from statsmodels.api import WLS
from scipy import stats
import matplotlib.pyplot as plt

#Change directory to XAI sub-folder of the project
os.chdir(r"F:\GW_LIME_VADER\XAI")


###Calculate optimal bandwidth (b) for GWR (Refer Section 3 of the paper to know about the bandwidth)

prepare_GWR_weights = False #Set to True to recalculate optimal bandwidth

if prepare_GWR_weights == True:
    mex_migration = pd.read_csv("./data/mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df.csv")
    
    
    ##Add target variable (predictions from the GraphSAGE model, used as the ground truth value, refer to Section 4.2, paragraph 1 of the paper)
    with open("./data/predictions_target_with_crime_no_implicit_target_num_migrants.txt", 'r') as file:
        predictions = ast.literal_eval(file.read())
        
    
    
    target = 'sum_num_initmig' 
    predictions = pd.DataFrame(predictions, columns = [target])
    
    mex_migration = pd.merge(mex_migration, predictions, left_index = True, right_index = True, how = 'left')
    
    
    
    #Execute Principal Component Extraction to avoid issues due to multicollinearity while calculating optimal bandwidth for GWR (See Section 4.1 of the paper)
    all_columns = list(mex_migration.columns) #store a list of all columns in the dataset
    predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', target, 'Latitude', 'Longitude']] #exclude columns that are not in the expalanatory variable set
    predictor_dataframe = mex_migration[predictor_columns]
    
    
    predictor_dataframe = scaler.fit_transform(predictor_dataframe)
    
    
    #cross validation to select optimal PC's to retain was executed, 20 to retain was the ideal value
    
    n_components = 20
    
    pca = PCA(n_components=n_components)
    
    # Fit and transform the scaled data using PCA
    principal_components = pca.fit_transform(predictor_dataframe)
    
    #Convert to a dataframe for easier analysis
    
    column_names_pcs = []
    
    for i in range(n_components):
        column_names_pcs.append('PC_' + str(i+1))
    
    
    
    principal_components_df = pd.DataFrame(principal_components, columns = column_names_pcs)
    
    mex_dataframe_principal_components = pd.merge(mex_migration[['GEO2_MX', target, 'Latitude', 'Longitude']], principal_components_df, left_index = True, right_index = True, how = 'left')
    
    u = mex_dataframe_principal_components['Longitude']
    v = mex_dataframe_principal_components['Latitude']
    coords = list(zip(u,v))
    
    #prepare data in the required format for extracting neighborhood weights
    
    all_columns = list(mex_dataframe_principal_components.columns) #store a list of all columns in the dataset
    
    predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', target, 'Latitude', 'Longitude']]
    predictor_dataframe = mex_dataframe_principal_components[predictor_columns]
    
    #scale the data
    #predictor_dataframe = scaler.fit_transform(predictor_dataframe)
    #X = predictor_dataframe.copy()

    X = predictor_dataframe.values
    
    
    predicted_dataframe = mex_dataframe_principal_components[[target]] 
    
    
    #change dtypes for code compatibility
    y = predicted_dataframe.values.reshape((-1,1)) # reshape is needed to have column array
    
    gwr_selector = Sel_BW(coords, y, X, spherical= True)
    gwr_bw = gwr_selector.search()
    print('GWR bandwidth =', gwr_bw) #Optimal GWR bandwidth = 151.0
    
    
    #Fit a GWR model to extract the weights (adaptive bisquare, golden selection rule, refer to Section 4.1 of the Paper)
    gwr_results = GWR(np.array(coords), np.array(y), np.array(X), gwr_bw).fit()
    
    
    weights_gwr = gwr_results.W
    
    weights_array = []
    
    for item in weights_gwr:
        weights_array.append(list(item))
        
    
    gwr_weights_df = pd.DataFrame(columns = ['GEO2_MX', 'weights'])
    
    
    for i in range(len(mex_dataframe_principal_components)):
        gwr_weights_df.loc[i, 'GEO2_MX' ] = mex_dataframe_principal_components.loc[i, 'GEO2_MX']
        gwr_weights_df.loc[i, 'weights' ] = weights_array[i]
    
    #export weights per municipality for use later in the XAI related Experiments
    
    gwr_weights_df.to_csv('./data/GWR_Weights_adaptive_bisquare_target_num_migrants_crime_variables_no_implicit_PCA.csv', index= False)


##Execute XAI computations (LIME and GW-LIME)


#Import dataset (Dataframe format of the feature matrix M', refer to Section 4.1 of the Paper)

mex_migration = pd.read_csv("./data/mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df.csv")


#Add target variable (predictions from the GraphSAGE model, used as the ground truth value, refer to Section 4.2, paragraph 1 of the paper)
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


#standardize the predictor variable dataframe

predictor_dataframe = scaler.fit_transform(predictor_dataframe)

X = predictor_dataframe.copy()

predicted_dataframe = mex_migration_GWR_LIME[[target]] 

y = predicted_dataframe.values.reshape((-1,1)) # reshape is needed to have column array

#import neighborhood weights calculated using GWR earlier in this code

gwr_weights_df = pd.read_csv("./data/GWR_Weights_adaptive_bisquare_target_num_migrants_crime_variables_no_implicit_PCA.csv")

gwr_weights_df['weights'] = gwr_weights_df['weights'].apply(ast.literal_eval)

gwr_weights_df['GEO2_MX'] = gwr_weights_df['GEO2_MX'].astype(str)


#Function to execute K-Lasso, and return the given values: k most important features, local R squared, local MAE, local MAPE (Refer to Section 3 for more on K-Lasso, and Section 4.2 for local R-sqared and MAE) 
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
    mape = np.mean(np.abs((residuals) / (y+ 1)) * instance_of_interest_neighborhood_wts) * 100 #y+1 to avoid 0 division

    
    
    instance_predicted_value = predictions[instance_index]
    
    
    
    #get rmse
    mse = np.mean(weighted_residuals ** 2)

    ##Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)    
    
    
    if return_only_performance_metrics == False:
        explanations_df = pd.DataFrame(columns = ['feature_name', 'beta_coefficient', 'p-value'])
        
        for i in range(len(used_features)):
            feature_index = used_features[i]
            predictor_name = predictor_columns[feature_index]
            beta_coefficient = beta_coefficients[i+1]
            row = {'feature_name' : predictor_name, 'beta_coefficient' : beta_coefficient, 'p-value': significance_values[i+1]}
            explanations_df = pd.concat([explanations_df, pd.DataFrame([row])], ignore_index=True)

        
        explanations_df = explanations_df.reindex(explanations_df['beta_coefficient'].abs().sort_values(ascending=False).index)

    else:
        explanations_df = 'Null'
        
    return [explanations_df, adjusted_r_squared, rmse, mae, instance_residual, mape, residuals, weighted_residuals, instance_predicted_value]


#####Execute function defined above to extract explanations and performance metrics for a given municipality#########

#explanation_yucatan = gwr_lasso_explain_instance('484004002', 10, return_only_performance_metrics = True)[1:4]
#explanation_sinaloa = gwr_lasso_explain_instance('484010034', 10, return_only_performance_metrics = False)[0]#484025003


##For every municipality, for GW-LIME, return and store performance metrics values for local fidelity experiment (Refer Section 4.2 of the paper)

os.chdir(r'./data/Explanations and Performance Indicatiors/GWR') #directory that will hold the results


list_of_munis = list(mex_migration_GWR_LIME['GEO2_MX'])

performance_indicatiors_GWR_LIME = pd.DataFrame(columns = ['GEO2_MX', 'adjusted_r_squared', 'rmse', 'mae', 'residual', 'mape'])


for item in list_of_munis:
    [local_r_squared, rmse, mae, residuals, mape] = gwr_lasso_explain_instance(item, 10, return_only_performance_metrics = True)[1:6]
    row = {'GEO2_MX' : item, 'adjusted_r_squared' : local_r_squared,'rmse' : rmse, 'mae' : mae, 'residual': residuals, 'mape': mape}
    performance_indicatiors_GWR_LIME = pd.concat([performance_indicatiors_GWR_LIME, pd.DataFrame([row])], ignore_index=True)
    

performance_indicatiors_GWR_LIME.to_csv("all_performance_indicators_GWR_LIME.csv", index = False)


#Local Fidelity Experiment: Calculate median values of performance metrics (Refer Section 4.2 of the paper)

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


#For each municipality, extract top 10 most important features for critical inferential tasks and case studies (See Section 5.2 for sample Case Studies)
os.chdir(r"./top_k_important_features")

list_of_munis = list(mex_migration_GWR_LIME['GEO2_MX'])

for item in list_of_munis:
    explanation= gwr_lasso_explain_instance(item, 10, return_only_performance_metrics = False)[0]
    important_features = list(explanation['feature_name'])
    explanation.to_csv(f'GWR_LIME_explanation_{item}.csv', index = False)




###Model 2: Baseline LIME
#Import the LIME package (Note: Original LIME code was edited to not generate preturbed dataset. Please read the LIME package code and related README)
import lime
import lime.lime_tabular


#Data Preprocessing

mex_migration_OLS_LIME = mex_migration.copy()

mex_migration_OLS_LIME['GEO2_MX'] = mex_migration_OLS_LIME['GEO2_MX'].astype(str)

mex_migration_OLS_LIME = mex_migration_OLS_LIME.fillna(0)

all_columns = list(mex_migration_OLS_LIME.columns)

predictor_columns = [i for i in all_columns if i not in ['GEO2_MX', 'Latitude', 'Longitude', target]]

predictor_dataframe = mex_migration_OLS_LIME[predictor_columns]

predicted_dataframe = mex_migration_OLS_LIME[['GEO2_MX', target]]
predicted_dataframe['GEO2_MX'] = predicted_dataframe['GEO2_MX'].astype(str)


#define predict function for LIME

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


#Function to generate expalations for a given municipality
def OLS_LIME_Instance_Explainer(instance_to_explain, inverse_array, no_of_features):
    exp = explainer.explain_instance(instance_to_explain, predict_fn = predict_migration, num_features=no_of_features, num_samples = len(training_dataset), list_of_prediction_id = inverse_array)
    #print(exp.as_list())
    
    return([exp.as_list(), exp.score, list(exp.weights), exp.predicted_values])


#Function to select the instance of interest

def OLS_LIME_Instance_Selector(explanatory_variables_df, explained_variable_df, instance_to_explain, number_of_features_k):
    id_of_interest = instance_to_explain
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




#Function to execute K-Lasso, and return the given values: k most important features, local R squared, local MAE, local MAPE (Refer to Section 3 for more on K-Lasso, and Section 4.2 for local R-sqared and MAE) 
 
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
#ols_results_aguacalientes  = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, '484001002', 10, return_only_performance_metrics = False)[4]



##For every municipality, for LIME, return and store performance metrics values for local fidelity experiment (Refer Section 4.2 of the paper)
os.chdir(r"F:\GW_LIME_VADER\XAI\data\Explanations and Performance Indicatiors\OLS")


list_of_munis = list(mex_migration_OLS_LIME['GEO2_MX'])


performance_indicatiors_OLS_LIME = pd.DataFrame(columns = ['GEO2_MX', 'adjusted_r_squared', 'rmse', 'mae', 'residual', 'mape'])


for item in list_of_munis:
    [local_r_squared, rmse, mae, residual, mape] = OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, item, 10, return_only_performance_metrics = True ) [0:5]
    row = {'GEO2_MX' : item, 'adjusted_r_squared' : local_r_squared,'rmse' : rmse, 'mae': mae, 'residual': residual, 'mape' : mape }
    performance_indicatiors_OLS_LIME = pd.concat([performance_indicatiors_OLS_LIME, pd.DataFrame([row])], ignore_index=True)
    

performance_indicatiors_OLS_LIME.to_csv("all_performance_indicators_OLS_LIME.csv", index = False)

####

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



summary_df_adj_rsquared_OLS_LIME.to_csv('adj_r_squared_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_rmse_OLS_LIME.to_csv('rmse_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_mae_OLS_LIME.to_csv('mae_performance_metrics_summary_OLS_LIME.csv', index = False)
summary_df_mape_OLS_LIME.to_csv('mape_performance_metrics_summary_OLS_LIME.csv', index = False)


#For each municipality, for baseline LIME, extract top 10 most important features for critical inferential tasks and case studies (See Section 5.2 for sample Case Studies)

os.chdir(r".\top_k_important_features")
list_of_munis = list(mex_migration_OLS_LIME['GEO2_MX'])

for item in list_of_munis:
    explanation= OLS_LIME_Explainer(predictor_dataframe, predicted_dataframe, item, 10, return_only_performance_metrics = False)[4]
    explanation.to_csv(f'OLS_LIME_explanation_{item}.csv', index = False)


#####Moran's I of Residuals, see Section 4.2 for more detials about the Experiment########

#import residuals for GW-LIME

gwr_lime_residuals = pd.read_csv(r"F:\GW_LIME_VADER\XAI\data\Explanations and Performance Indicatiors\GWR\all_performance_indicators_GWR_LIME.csv")

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

#import residuals for baseline LIME

ols_lime_residuals = pd.read_csv(r"F:\GW_LIME_VADER\XAI\data\Explanations and Performance Indicatiors\OLS\all_performance_indicators_OLS_LIME.csv")

ols_lime_residuals = ols_lime_residuals[['GEO2_MX', 'residual']]

ols_lime_residuals_array = np.array(ols_lime_residuals['residual'])

moran_2 = Moran(ols_lime_residuals_array, w_knn)



print("Moran's I:", moran_2.I)
print("Expected Moran's I:", moran_2.EI)
print("p-value:", moran_2.p_norm)
print("z-score:", moran_2.z_norm)



