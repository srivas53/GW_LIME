This folder contains all the necessary code to train the black-box GraphSAGE model. All code was written by co-author Heather Baier. The scripts listed below should be executed in sequence:

1. construct_graph: Constructs the graph to be used for training the GraphSAGE model. Also outputs the tabular dataset for use in the XAI models.
   
2. train_GraphSAGE: Trains the GraphSAGE model. Outputs a `.torch` file with the optimal learned parameters.
   
3. generate_predictions_forward_pass: Uses the trained `.torch` file to generate forward pass predictions for the number of migrants per municipality. Outputs a `.txt` file with the predictions to be used in the XAI stage.

Supporting Libraries :
- GeoGraph
- encoder
- graphsage