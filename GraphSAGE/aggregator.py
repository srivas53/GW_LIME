import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda = False, gcn = False): 
        
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample = 4):
        
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """

        # Local pointers to functions (speed hack)
        _set = set
        
        # If nodes are to be sampled, do so
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + list(set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]  
                    
        unique_nodes_list = list(set([item for sublist in samp_neighs for item in sublist]))    
        
        # Give each node and index from 0 to len(unique_nodes_list)
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}        
        
        # Create an emtpy mask that essentially serves as an adjacency matrix
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))      
        
        # Grab the column & row indices (0 -> n) and insert into the adjacnency matrix
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]      
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]        
        mask[row_indices, column_indices] = 1
                
        if self.cuda:
            mask = mask.cuda() 
            
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        unique_nodes_list_int = torch.tensor(list(set([int(item) for sublist in samp_neighs for item in sublist])))
                        
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list_int).cuda())
        else:
            embed_matrix = torch.index_select(torch.tensor(self.features, dtype = torch.float32), 0, unique_nodes_list_int)
            
            
#         print("EMBED MATRIX: ")
        
#         print(embed_matrix.shape)
                        
        to_feats = mask.mm(embed_matrix)
        
#         print("TO FEATS: ")
        
#         print(to_feats.shape)
                
        # Replace all of the null values where there are no neighbors with 0
        to_feats[to_feats != to_feats] = 0

        return to_feats
