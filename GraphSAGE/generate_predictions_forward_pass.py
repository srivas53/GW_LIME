# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:15:31 2023

@author: vsriva11
"""

import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import json
import os
os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME\GraphSAGE")

from aggregator import *
from graphsage import *
from encoder import *
import geopandas as gpd

#Import trained GraphSAGE model torch file

graph_checkpoint = torch.load("./trained_GraphSAGE_model.torch")
best_validation_mae = graph_checkpoint["best_mae"]

graph_checkpoint = graph_checkpoint["model_state_dict"]


#Import graph used for training
with open("./data/graph_with_crime_no_implicit_target_num_migrants.json") as g:
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

#y = y.tolist()

#calculate evaluation metrics for the forward pass
from sklearn.metrics import mean_absolute_error, r2_score
r2 = r2_score(y, predictions)
mae =   mean_absolute_error(y, predictions)

#export predictions to be used in XAI procedures as the ground truth (refer to section 3.3.3, paragraph 1)


predictions = str(predictions)

with open(r'D:\vsriva11\VADER Lab\GW_LIME\XAI\data\predictions_target_with_crime_no_implicit_target_num_migrants.txt', 'w') as file:
    file.write(predictions)


#create a dataframe that has predictions and original labels together, referenced by GEO2_MX
# graph_id_to_geomx = pd.read_csv(r"D:\vsriva11\VADER Lab\graph_stuff-20230928T171307Z-001\graph_stuff\shapeID_to_GEOMX_Map.csv")

# predictions_df = pd.DataFrame(predictions)
# original_labels_df = pd.DataFrame(y)

# graph_id_to_geomx = pd.merge(graph_id_to_geomx, predictions_df, left_index=  True, right_index = True, how = 'left')
# graph_id_to_geomx = pd.merge(graph_id_to_geomx, original_labels_df, left_index=  True, right_index = True, how = 'left')

# #df.rename(columns={'A': 'Column1', 'B': 'Column2'})
# graph_id_to_geomx = graph_id_to_geomx.rename(columns = {'GEO2_MX' : 'GEO2_MX', 'ShapeID' : 'ShapeID', '0_x' : 'predicted_value', '0_y':'ground_truth'})

# graph_id_to_geomx.to_csv('predictions_and_ground_truth_target_with_crime_no_implicit_target_perc_migrants.csv', index= False)



#predictions in negative will be converted to zero

#R squared: 0.7460129357916963 with original variables, target sum_num_initmig
#R squared: 0.7483954669971528 with original variables, target perc migrants
#R squared: 0.6999091387785561 with only subset of socio-economic variables(ones displayed on migration portal), target perc migrants
#mae: 2.433550449494402 with original variables, target perc migrants
#mae: 2.660022841664129 with subset of socio-economic variables, target perc migrants

























