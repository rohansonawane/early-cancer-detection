import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class SimilarityNetworkFusion:
    def __init__(self, K=20, t=20):
        """
        Initialize SNF
        
        Args:
            K (int): Number of nearest neighbors
            t (int): Number of iterations
        """
        self.K = K
        self.t = t
        
    def _compute_similarity(self, data):
        """Compute similarity matrix using RBF kernel"""
        dist = squareform(pdist(data))
        sigma = np.mean(dist) * 0.5
        W = np.exp(-dist**2 / (2 * sigma**2))
        return W
    
    def _normalize_adjacency(self, W):
        """Normalize adjacency matrix"""
        D = np.sum(W, axis=1)
        D = np.diag(1.0 / np.sqrt(D))
        W_norm = D @ W @ D
        return W_norm
    
    def _compute_affinity_matrix(self, W):
        """Compute affinity matrix using K nearest neighbors"""
        n = W.shape[0]
        W_knn = np.zeros_like(W)
        
        for i in range(n):
            idx = np.argsort(W[i, :])[-self.K:]
            W_knn[i, idx] = W[i, idx]
            
        W_knn = (W_knn + W_knn.T) / 2
        return W_knn
    
    def fuse(self, data_list):
        """
        Fuse multiple similarity networks
        
        Args:
            data_list (list): List of data matrices for different omics types
            
        Returns:
            W_fused: Fused similarity network
        """
        n = data_list[0].shape[0]
        W_list = []
        
        # Compute similarity matrices for each omics type
        for data in data_list:
            W = self._compute_similarity(data)
            W = self._normalize_adjacency(W)
            W = self._compute_affinity_matrix(W)
            W_list.append(W)
        
        # Initialize fused network
        W_fused = np.mean(W_list, axis=0)
        
        # Iterative fusion
        for _ in range(self.t):
            W_new = np.zeros_like(W_fused)
            for W in W_list:
                W_new += W @ W_fused @ W.T
            W_fused = W_new / len(W_list)
            W_fused = self._normalize_adjacency(W_fused)
            
        return W_fused

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) + self.bias
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class CrossOmicsGCN(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, dropout=0.5):
        super(CrossOmicsGCN, self).__init__()
        self.snf = SimilarityNetworkFusion()
        self.gcn_list = nn.ModuleList([
            GCN(dim, hidden_dim, hidden_dim, dropout)
            for dim in input_dims
        ])
        self.fusion = nn.Linear(hidden_dim * len(input_dims), output_dim)
        
    def forward(self, x_list, adj_list):
        # Fuse similarity networks
        adj_fused = self.snf.fuse(adj_list)
        adj_fused = torch.FloatTensor(adj_fused).to(x_list[0].device)
        
        # Process each omics type through GCN
        h_list = []
        for i, (x, gcn) in enumerate(zip(x_list, self.gcn_list)):
            h = gcn(x, adj_fused)
            h_list.append(h)
            
        # Concatenate and fuse features
        h_concat = torch.cat(h_list, dim=1)
        output = self.fusion(h_concat)
        
        return output 