# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:23:40 2023

@author: vsriva11
"""
import os
from sklearn import preprocessing
import geopandas as gpd
import pandas as pd
import numpy as np
from GeoGraph import *
import json
from ast import literal_eval
import ast
import sys

#Note: Please refer to Section 4.2, and Figure 7 of the paper for details about the senstivity analysis experiment

##Import the original feature matrix M
with open("/home/vsriva11/features.json") as fd:
    feat_data = json.load(fd) 


mig_data = pd.read_csv("/home/vsriva11/mexico2010_wcrime.csv")



##Data Preprocessing (filtering features)

#get predictor columns
#original variables
with open("/home/vsriva11/predictor_columns_original.txt", 'r') as file:
    predictor_columns = literal_eval(file.read())


target = 'sum_num_intmig' 'sum_num_intmig'

if target == 'perc_migrants':
    mig_data['perc_migrants'] = (mig_data['sum_num_intmig']/ mig_data['total_pop']) *100
    
    mig_data = mig_data.drop('sum_num_intmig', axis=1)


labels = {}
    
for key in feat_data:
    mig_data_target = mig_data[mig_data['GEO2_MX'] == int(key)] 
    mig_data_target = mig_data_target.reset_index(drop=True)
    labels[key] = mig_data_target.loc[0, target]



#remove implicit variables
predictor_columns_edited = []

for value in predictor_columns:
    new_value = value.replace('perc_yes_', '')
    final_value = new_value.replace('perc_no_', '')
    final_value = final_value.replace('_not_', '_')
    
    predictor_columns_edited.append(final_value)


def find_first_repeating_indices(data):
    value_indices = {}
    repeating_indices = []

    for index, value in enumerate(data):
        if value in value_indices and value not in repeating_indices:
            repeating_indices.append(value_indices[value])
        else:
            value_indices[value] = index

    return repeating_indices


first_repeating_indices = find_first_repeating_indices(predictor_columns_edited)
predictor_columns = [value for index, value in enumerate(predictor_columns) if index not in first_repeating_indices]


#Add crime variables
crime_variables = ['abduction_forced_disappearance',
'armed_clash',
'arrests',
'attack',
'change_to_group_activity',
'looting_property_destruction',
'sexual_violence']


predictor_columns.extend(crime_variables)

#Define an empty dataset that will record the prediction performance degradation of GraphLIME

performance_degradation_df = pd.DataFrame(columns = ['GEO2_MX', 'perc_increase_actual_set', 'perc_increase_random_set'])

ground_truth_and_predicted_df = pd.read_csv("/home/vsriva11/predictions_and_ground_truth_target_with_crime_no_implicit_target_perc_migrants.csv") 

list_of_shape_ids = list(ground_truth_and_predicted_df['GEO2_MX'])
instance_of_interest = list_of_shape_ids[int(sys.argv[1])]


    
ground_truth_and_predicted_df_subset = ground_truth_and_predicted_df[ground_truth_and_predicted_df['GEO2_MX'] == instance_of_interest]
ground_truth_and_predicted_df_subset = ground_truth_and_predicted_df_subset.reset_index(drop = True)


#Store the original absolute error for calculation performance degradation

ground_truth = ground_truth_and_predicted_df_subset.loc[0, 'ground_truth']

predicted_value_original = ground_truth_and_predicted_df_subset.loc[0, 'predicted_value']

original_absolute_error = abs(ground_truth-predicted_value_original)
instance_of_interest_index = ground_truth_and_predicted_df_subset.loc[0, 'ShapeID']



################Step 1: Graph generation, dropping the important features for the instance of interest accorgin to GW-LIME##############

important_features_gwr_lime = pd.read_csv(f"/home/vsriva11/GWR_LIME/GWR_LIME_explanation_{instance_of_interest}.csv")

important_features_gwr_lime = important_features_gwr_lime[['feature_name']]

important_features_gwr_lime = list(important_features_gwr_lime['feature_name'])


#Function to generate graph with values of top-k important features for the instance of interest with random noise (Refer Figure 7 of the Paper)

def generate_graph_important_features_removal(list_of_features_to_edit, method, randomize): #method : gwr, ols
    remove_satellite = 0 #keep staellite featues as is
    ##################################
    #update the list of features to edit to contain only non-satellite features
    
    list_of_features_to_edit = [i for i in list_of_features_to_edit if 'Satellite' not in i]
    # Define the number of noisy features to add
    num_noisy_features = len(list_of_features_to_edit)    
    
    #if randomize true, then draw 10 random features not in set S_k
    if randomize == True:
        draw_from_list = [i for i in predictor_columns if i not in list_of_features_to_edit]
        list_of_features_to_edit = random.sample(draw_from_list, num_noisy_features)            
    
    #generate a noise vector, drawn randomly from a Gaussian distribution


    # Define the number of samples (assumes the same number of samples as original_data)
    num_samples = len(mig_data)

    # Define the parameters for the noisy features
    noise_params = {
        'mean': 0,        # Mean of the Gaussian distribution
        'stddev': 0.2,    # Standard deviation of the Gaussian distribution
    }
    
    # Create a DataFrame for the noisy features
    noisy_features = pd.DataFrame()
    
    for i in range(num_noisy_features):
        # Generate noisy data for each feature
        noisy_data = np.random.normal(noise_params['mean'], noise_params['stddev'], num_samples)
        
        # Create a column name for the noisy feature (e.g., NoisyFeature1, NoisyFeature2, ...)
        col_name = list_of_features_to_edit[i]
        
        # Add the noisy feature to the noisy_features DataFrame
        noisy_features[col_name] = noisy_data

    predictor_columns_updated = [i for i in predictor_columns if i not in list_of_features_to_edit]
    
    
    predictor_dataframe = mig_data[predictor_columns_updated]

    #replace removed columns with noise
    predictor_dataframe = pd.concat([predictor_dataframe, noisy_features], axis=1)        
    
    new_keys = [str(i) for i in mig_data['GEO2_MX'].to_list()]
    new_vals = predictor_dataframe.values
    
    
    mMScale = preprocessing.MinMaxScaler()
    new_vals = mMScale.fit_transform(new_vals)
    
    census = dict(zip(new_keys, new_vals))
    
    keep_ids = list(feat_data.keys())
    
    
    gdf = gpd.read_file("/home/vsriva11/ipumns_shp.shp")
    # print(gdf.dtypes)
    gdf = gdf.dropna(subset = ["geometry"])
    gdf = gdf[gdf['shapeID'].isin(keep_ids)]

    
    
    print("Number of polygons: ", len(gdf), "  |  Number of ID's: ", len(gdf['shapeID'].to_list()))
    
    shapeIDs = gdf['shapeID'].to_list()
    gdf["shapeID"] = [str(i) for i in range(0, len(gdf))]
    
    
    graph = {}
    
    if remove_satellite == 0:
        for shapeID in gdf['shapeID'].to_list():  
            
            print(shapeID, end = "\r")
            
            muni_id = shapeIDs[int(shapeID)]
                    
            g = GeoGraph(str(shapeID),
                                gdf, 
                                degrees = 1, 
                                load_data = False, 
                                boxes = False)
            
            node_attrs = {'x': np.append(feat_data[muni_id], census[muni_id]).tolist(),
                    'label': labels[muni_id],
                    'neighbors': g.degree_dict[1]
                }
        
            graph[shapeID] = node_attrs
    else:
        for shapeID in gdf['shapeID'].to_list():  
            
            print(shapeID, end = "\r")
            
            muni_id = shapeIDs[int(shapeID)]
                    
            g = GeoGraph(str(shapeID),
                                gdf, 
                                degrees = 1, 
                                load_data = False, 
                                boxes = False)
            
            node_attrs = {'x': census[muni_id].tolist(),
                    'label': labels[muni_id],
                    'neighbors': g.degree_dict[1]
                }
        
            graph[shapeID] = node_attrs           
    
    
    
    #export the updated graph with random noise (Refer Figure 7 of the paper)
    with open(f"/home/vsriva11/Updated_graphs/graph_replace_with_noise_senstivity_analysis_randomize_{str(randomize)}_{method}_{instance_of_interest}.json", "w") as outfile: 
        json.dump(graph, outfile)




generate_graph_important_features_removal(important_features_gwr_lime, 'gwr', randomize = False) #Exports updated graph P_k
generate_graph_important_features_removal(important_features_gwr_lime, 'gwr', randomize = True) #Exports updated graph P_k'



#Step 2: generate prediction via forward pass, store absolute error for the instance of interest, and calculate performance degradation

import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import json

from aggregator import *
from graphsage import *
from encoder import *
import geopandas as gpd


def generate_predictions_store_absolute_error(instance_of_interest, method, randomize):
    graph_checkpoint = torch.load("/home/vsriva11/trained_graph_model_with_crime_no_implicit_target_num_migrants.torch")
    
    graph_checkpoint = graph_checkpoint["model_state_dict"]
    
    with open(f"/home/vsriva11/Updated_graphs/graph_replace_with_noise_senstivity_analysis_randomize_{str(randomize)}_{method}_{instance_of_interest}.json") as g:
        graph = json.load(g)
    
    
    x, adj_lists, y = [], {}, []
    
    a = 0
    for muni_id, dta in graph.items():
        x.append(dta["x"])
        y.append(dta["label"])
        adj_lists[str(a)] = dta["neighbors"]
        a += 1
        
    x = np.array(x)
    #y = np.expand_dims(np.array(y), 1)
    
    agg = MeanAggregator(features = x, gcn = False)
    enc = Encoder(features = x, feature_dim = x.shape[1], embed_dim = 128, aggregator = agg, adj_lists = adj_lists) #adj_lists = adj_lists
    model = SupervisedGraphSage(num_classes = 1, enc = enc)
    model.load_state_dict(graph_checkpoint)
    model.eval()
    
    
    predictions = []
    for i in range(len(graph)):
        try:
            #muni_ref = graph_id_dict[i]        
            input_1 = [i]
            #print('INPUT IS THIS', input)
            #print(model.forward(input))
            prediction = int(model.forward(input_1).item())
            if prediction <0:
                prediction = 0
                
            predictions.append(prediction)
        except:
            predictions.append(0)
    
    
    prediction_instance_of_interest = predictions[instance_of_interest_index]
    
    absolute_error_removed_features = abs(ground_truth - prediction_instance_of_interest)
    return absolute_error_removed_features


abs_error_actual_set = generate_predictions_store_absolute_error(instance_of_interest = instance_of_interest, method = 'gwr', randomize = False)
abs_error_random_set = generate_predictions_store_absolute_error(instance_of_interest = instance_of_interest, method = 'gwr', randomize = True)


perc_increase_actual_set =  ((abs_error_actual_set - original_absolute_error)/original_absolute_error)*100
perc_increase_random_set =  ((abs_error_random_set - original_absolute_error)/original_absolute_error)*100

row = {'GEO2_MX' : instance_of_interest, 'perc_increase_actual_set' : perc_increase_actual_set, 'perc_increase_random_set': perc_increase_random_set}
performance_degradation_df = pd.concat([performance_degradation_df, pd.DataFrame([row])], ignore_index=True)


performance_degradation_df.to_csv(f'/home/vsriva11/Senstivity_analysis_results/performance_degradation_senstivity_analysis_{instance_of_interest}.csv', index = False)


