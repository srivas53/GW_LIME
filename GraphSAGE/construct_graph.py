from sklearn import preprocessing
import geopandas as gpd
import pandas as pd
import numpy as np
import os
os.chdir(r"D:\vsriva11\VADER Lab\GW_LIME\GraphSAGE")
from GeoGraph import *
import json
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.decomposition import PCA



#import satellite features
with open("./data/features.json") as fd:
    feat_data = json.load(fd)  



#Import the original dataset containing 284 socio-economic and crime variables extracted from INEGI and ACLED
   
mig_data = pd.read_csv("./data/mexico2010_wcrime.csv")

#Prepare label data dictionary (to be used in graph generation process)

target = 'sum_num_intmig' #possible target variables: 'perc_migrants', 'sum_num_intmig'


if target == 'perc_migrants':
    mig_data['perc_migrants'] = (mig_data['sum_num_intmig']/ mig_data['total_pop']) *100
    
    mig_data = mig_data.drop('sum_num_intmig', axis=1)
   

labels = {}
    
for key in feat_data:
    mig_data_target = mig_data[mig_data['GEO2_MX'] == int(key)] 
    mig_data_target = mig_data_target.reset_index(drop=True)
    labels[key] = mig_data_target.loc[0, target]


#Feature Selection

#Step 1: Removing implicit variables (eg: yes_phone, no_phone) 
#get a list of all predictor variables related socio-economic indicators originally derived from INEGI in .txt format (easy to check for implicit variables)

with open("./data/predictor_columns_original.txt", 'r') as file:
    predictor_columns = literal_eval(file.read())


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



#Step 2: Select the crime varaibles to be included in the GraphSAGe training

crime_variables = ['abduction_forced_disappearance',
 'armed_clash',
 'arrests',
 'attack',
 'change_to_group_activity',
 'looting_property_destruction',
 'sexual_violence']


predictor_columns.extend(crime_variables)


            

#Filter the original dataset to include only the non implicit socio-economic variables and a subset of crime variables

predictor_dataframe = mig_data[predictor_columns]


#Initiate data preparation for graph construction

new_keys = [str(i) for i in mig_data['GEO2_MX'].to_list()]
new_vals = predictor_dataframe.values

#Normalizing the quantitavie values

mMScale = preprocessing.MinMaxScaler()
new_vals = mMScale.fit_transform(new_vals)

census = dict(zip(new_keys, new_vals))

#Remove municiplaites that have insufficient information due to data errors

keep_ids = list(feat_data.keys())

#Import shape file to determine adjacent municiplaities (critical for defining adjacency matrix)
gdf = gpd.read_file("./data/ipumns_shp.shp")
gdf = gdf.dropna(subset = ["geometry"])
gdf = gdf[gdf['shapeID'].isin(keep_ids)]
centroids = gdf.copy()


print("Number of polygons: ", len(gdf), "  |  Number of ID's: ", len(gdf['shapeID'].to_list()))

shapeIDs = gdf['shapeID'].to_list()


gdf["shapeID"] = [str(i) for i in range(0, len(gdf))]

#Before generating the graph, we will prepare the dataset used to execute the XAI methods (Matrix M`, refer to section 3.3.1 of the paper)

predictor_columns_plus_primary_key = predictor_columns.copy() + ['GEO2_MX']

explanation_df = mig_data[predictor_columns_plus_primary_key]

explanation_df['GEO2_MX'] = explanation_df['GEO2_MX'].astype(str)

explanation_df = explanation_df[explanation_df['GEO2_MX'].isin(keep_ids)]




#add latitude and longitude of centroids for geo-referencing and other computations

centroids['geometry'] = centroids['geometry'].centroid
centroids = centroids[centroids['geometry'].is_empty == False]
centroids = centroids.reset_index(drop = True)
centroids['Longitude'] = ''
centroids['Latitude'] = ''


for i in range(len(centroids)):
    centroids.loc[i, 'Longitude'] = centroids.loc[i,'geometry'].x
    centroids.loc[i, 'Latitude'] = centroids.loc[i,'geometry'].y


centroids = centroids.rename({'shapeID': 'GEO2_MX'}, axis = 1)
#merge with explainer dataframe
explanation_df = pd.merge(explanation_df, centroids[['GEO2_MX', 'Longitude', 'Latitude']], on = 'GEO2_MX', how = 'left')

##extract top three satellite principal components

satellite_df = pd.DataFrame(feat_data)
satellite_df = satellite_df.T
satellite_df['GEO2_MX'] = satellite_df.index

satellite_df = satellite_df.reset_index(drop = True)


columns_for_pc = [i for i in list(satellite_df) if i not in ['GEO2_MX']]

satellite_pcs = satellite_df[columns_for_pc]

satellite_pcs = scaler.fit_transform(satellite_pcs)

n_components = 3
pca = PCA(n_components=n_components)

# Fit and transform the scaled data using PCA
satellite_principal_components = pca.fit_transform(satellite_pcs)

column_names_pcs = []

for i in range(n_components):
    column_names_pcs.append('Satellite_PC_' + str(i+1))

satellite_principal_components = pd.DataFrame(satellite_principal_components, columns = column_names_pcs)


satellite_principal_components = pd.merge(satellite_df[['GEO2_MX']], satellite_principal_components, left_index = True, right_index = True, how = 'left')


explanation_df = pd.merge(explanation_df, satellite_principal_components, on = 'GEO2_MX', how = 'left' )

#Export the datasset to be used for XAI computations

explanation_df.to_csv(r'D:\vsriva11\VADER Lab\GW_LIME\XAI\data\mexico_migration_explanation_input_with_crime_no_implicit_target_num_migrants_df.csv', index = False)





#Generate graph to be used for GraphSAGE training

graph = {}

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



with open("./data/graph_with_crime_no_implicit_target_num_migrants.json", "w") as outfile: 
    json.dump(graph, outfile)









    