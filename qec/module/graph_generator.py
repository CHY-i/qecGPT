import numpy as np
import networkx as nx
import torch
import os

def generate_graph(n, m, degree=3, seed=0,  G_type='rrg', random_weights=False):
    n_nodes = n * m
    torch.manual_seed(seed)
    if G_type == 'rrg': 
        G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    elif G_type == 'erg':
        G = nx.erdos_renyi_graph(n, degree/n, seed=seed)
    elif G_type == '2D':
        M = nx.grid_graph([n, m])
        A = np.array(nx.adjacency_matrix(M).todense())
        G = nx.from_numpy_array(A)
    
    J = np.array(nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes())).todense())
    J = torch.from_numpy(J).to(torch.float64)
    if random_weights==False:
        J = J
    elif random_weights==True:
        W = torch.randn_like(J)
        W = (W + W.T)/2
        J = J*W
    elif random_weights=='random_one':
        W = torch.triu(J)
        W = torch.bernoulli(W/2)*2-W
        J = W + W.T
    return G, J

