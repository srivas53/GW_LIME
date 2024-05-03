This folder contains all the necessary code to train the black-box GraphSAGE model. All code authored by Heather Baier. The scripts listed below are to be executed in sequence:

1. construct_graph: Script that constructs graph to be used for training GraphSAGE model. Also outputs the tabular dataset to be used in the XAI models.
2. train_GraphSAGE: Script used to train ther GraphSAGE model. Outputs a .torch file with the optimal learned parameters. 
3. generate_predictions_forward_pass: Takes the trained .torch file and generates forward pass predictions for number of muigrants per municipality. Outputs a .txt file with the predictionms to be used in the XAI stage.

Supporting Libraries (autoored by Heather Baier):
1. GeoGraph, encoder, graphsage
