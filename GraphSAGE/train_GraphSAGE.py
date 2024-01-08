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


for it in range(0, 1):
    
    print("IT: ", it)

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
    y = np.expand_dims(np.array(y), 1)

    agg = MeanAggregator(features = x,
                        gcn = False)
    enc = Encoder(features = x, 
                  feature_dim = x.shape[1], 
                  embed_dim = 128, 
                  adj_lists = adj_lists,
                  aggregator = agg)

    model = SupervisedGraphSage(num_classes = 1, enc = enc)

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01)

    n = 64 # batch_size

    train_num = int(x.shape[0] * .70)
    train_indices = random.sample(range(0, x.shape[0]), train_num)
    val_indices = [i for i in range(0, x.shape[0]) if i not in train_indices]

    train_indices_b = [train_indices[i * n:(i + 1) * n] for i in range((len(train_indices) + n - 1) // n )] 
    val_indices_b   = [val_indices[i * n:(i + 1) * n] for i in range((len(val_indices) + n - 1) // n )] 
    

    best_mae = 9000000000000
    best_weights = model.state_dict()
    epoch_best_wts = 0
    for epoch in range(0, 1000):

        running_train_loss, running_val_loss = 0, 0

        for batch in train_indices_b:

            model.train()

            batch_nodes = [str(i) for i in batch]
            batch_ys = torch.tensor([y[int(i)] for i in batch])

            optimizer.zero_grad()
            loss = model.loss(batch_nodes, batch_ys)
    #         
    #         print(loss)
    #         
    #         sfss
    #         
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        for batch in val_indices_b:

            model.eval()

            batch_nodes = [str(i) for i in batch]
            batch_ys = torch.tensor([y[int(i)] for i in batch])

            loss = model.loss(batch_nodes, batch_ys)
            running_val_loss += loss.item()

        t_loss = running_train_loss / len(train_indices_b)
        v_loss = running_val_loss / len(val_indices_b)

        print("Epoch: ", epoch, "Training Loss: ", t_loss, " Validation Loss: ", v_loss)
        
        ###############VS Add: for r-squared calculation################
        # Calculate R-squared for validation set
        # y_true = []
        # y_pred = []
        
        # for batch in val_indices_b:
        #     model.eval()
        #     batch_nodes = [str(i) for i in batch]
        #     batch_ys = torch.tensor([y[int(i)] for i in batch])
        
        #     # Assuming your model has a prediction method
        #     predictions = model.predict(batch_nodes)
        
        #     y_true.extend(batch_ys.numpy())
        #     y_pred.extend(predictions.numpy())
        
        # sse = np.sum((np.array(y_true) - np.array(y_pred))**2)
        # sst = np.sum((np.array(y_true) - np.mean(y_true))**2)
        
        # v_r_squared = 1 - (sse / sst)
                
        # print("Epoch: ", epoch, "Training Loss: ", t_loss, " Validation r_squared: ", v_r_squared)
        
        #########################################################################
        
        

        if v_loss < best_mae:
            best_weights = model.state_dict()
            best_mae = v_loss
            epoch_best_wts = epoch
            
            print("New Best MAE at epoch", epoch)    


    torch.save({
                'epoch': epoch_best_wts,
                'best_mae' : best_mae,
                'model_state_dict': best_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                }, "./trained_GraphSAGE_model.torch")


    model.load_state_dict(best_weights)


    # Validate
    trues, preds, tv, indices = [], [], [], []

    for index in val_indices:

        try:

            input = [str(index)]
            output = torch.tensor(y[index])

            model.eval()

            loss = model.loss(input, output)

            trues.append(y[index][0])
            preds.append(model.scores.item())

            tv.append('val')
            
            indices.append(index)

        except:

            print(index)

    # for index in train_indices:

    #     try:

    #         input = [str(index)]
    #         output = torch.tensor(y[index])

    #         model.eval()

    #         loss = model.loss(input, output)

    #         trues.append(y[index][0])
    #         preds.append(model.scores.item())

    #         tv.append('train')
            
    #         indices.append(index)

    #     except:

    #         print(index)


    preds_df = pd.DataFrame()
    preds_df['muni_index'] = indices
    preds_df['true'], preds_df['pred'] = trues, preds
    #figure out R squared value
    from sklearn.metrics import mean_absolute_error, r2_score
    r2 = r2_score(trues, preds)
    
#     preds_df["abs_diff"] = abs(preds_df['true'] - preds_df['pred'])
#     preds_df['tv'] = tv
#     preds_df
    
# #     fname = "./results/preds_it" + str(it) + ".csv"

#     fname = "./results/preds_wcrime_subevent.csv"

#     preds_df.to_csv(fname)
    