import torch
import numpy as np
import networkx as nx
import opt_einsum as oe
import time
import random
import os

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def bench(mat, func, device):
        for i in range(10): # preheating
            func(mat)
        torch.cuda.synchronize(device)
        t0 = time.time()
        for i in range(10):
            func(mat)
        torch.cuda.synchronize(device)
        t1 = time.time()
        return (t1-t0)/10

def generate_graph(n, m=1, l=1, degree=3, seed=0,  G_type='2D', coupling='fe'):
    n_nodes = n * m * l
    
    set_seed(seed)

    if G_type == 'rrg': 
        G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    elif G_type == 'erg':
        G = nx.erdos_renyi_graph(n, degree/n, seed=seed)
    elif G_type == '1D':
        G = nx.path_graph(n_nodes)
    elif G_type == '2D':
        M = nx.grid_graph([n, m])
        A = np.array(nx.adjacency_matrix(M).todense())
        G = nx.from_numpy_array(A)
    elif G_type == '3D':
        M = nx.grid_graph([n, m, l])
        A = np.array(nx.adjacency_matrix(M).todense())
        G = nx.from_numpy_array(A)
    

    J = np.array(nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes())).todense())
    J = torch.from_numpy(J).to(torch.float32)
    if coupling=='fe':
        None
    elif coupling=='anti':
        J = -J
    elif coupling=='nor':
        W = torch.randn_like(J)
        W = (W + W.T)/2
        J = J*W
    elif coupling=='ao':
        W = torch.triu(J)
        W = torch.bernoulli(W/2)*2-W
        J = W + W.T
    elif coupling=='uni':
        W = torch.rand_like(J)
        W = (W + W.T)/2
        J = J*W
        #print(J)
    return G, J

class RandomIsing():
    def __init__(self, n, degree, coupling='fe', seed=36485, dtype=torch.float64, device='cpu'):
        self.n = n
        self.dtype=dtype
        self.device=device
        self.coupling = coupling

    
        self.graph, self.J = generate_graph(n=n, degree=degree, coupling=coupling, seed=seed, G_type='rrg')
        #print(self.J)
        self.get_bonds()
        
    def get_bonds(self):
        G = self.graph
        neighbors = []
        for i in range(self.n):
            a = list(G.neighbors(i))
            a.sort()
            neighbors.append(a)
        edges = []
        for i in range(len(neighbors)):
            for node in neighbors[i]:
                if node > i:
                    edges.append([i, node])
        nodes_bonds = []
        for i in range(self.n):
            nodes_bonds.append([])
            for j in range(len(edges)):
                bond = edges[j]
                if i in bond:
                    nodes_bonds[i].append(edges.index(bond))
        '''neighbors of physical site'''
        self.neighbors = neighbors 
        '''sites of edges'''
        self.edges = edges
        '''edges of sites'''
        self.nodes_bonds = nodes_bonds
    
    def MH(self, n_s, beta, s_step=500, h=0.1):
        s = []
        interval = self.n
        state = (torch.bernoulli(torch.tensor([0.5]*self.n))*2-1)
        #print(state)
        for i in range(s_step+n_s*interval):
            alpha = torch.rand(1)
            idx = torch.randint(0, self.n, (1, ))
            neighbors = self.neighbors[idx]
            delta_energy = 2*state[idx]*self.J[idx, neighbors] @ state[neighbors].T + 2*h*state[idx]
            #print(delta_energy)
            #print(alpha, torch.exp(-self.beta*delta_energy))
            if delta_energy <= 0 or alpha<torch.exp(-beta*delta_energy):
                state[idx] = -state[idx]
            else:
                None
            #print(state)
            if i+1 >s_step and (i+1)%interval ==0:
                s.append(state.tolist())
        s = torch.tensor(s)
        #print(s)
        return s


class Ising():
    def __init__(self, L, beta, dim=2, coupling='fe', seed=0, dtype=torch.float64, device='cpu'):
        self.L = L
        self.n = L**dim
        self.beta = beta
        self.dtype=dtype
        self.device=device
        self.coupling = coupling

        if dim == 1:
            self.graph, self.J = generate_graph(L, coupling=coupling, seed=seed, G_type='1D')
        elif dim == 2:
            self.graph, self.J = generate_graph(L, L, coupling=coupling, seed=seed, G_type='2D')
        elif dim == 3:
            self.graph, self.J = generate_graph(L, L, L, coupling=coupling, seed=seed, G_type='3D')
        
        self.get_bonds()
    
    
    def energy(self, sample, h=0):
        J = self.J.to(self.device).to(self.dtype)
        h = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        batch = sample.shape[0]
        D = sample.shape[1]
        J = J.to_sparse()
        energy = - torch.bmm(sample.view(batch, 1, D),
                             torch.sparse.mm(J, sample.t()).t().view(batch, D, 1)).reshape(batch) / 2 - sample @ h

        return energy
    def MH(self, n_s, s_step=500, h=0.1):
        s = []
        interval = self.n
        state = (torch.bernoulli(torch.tensor([0.5]*self.n))*2-1)
        #print(state)
        for i in range(s_step+n_s*interval):
            alpha = torch.rand(1)
            idx = torch.randint(0, self.n, (1, ))
            neighbors = self.neighbors[idx]
            delta_energy = 2*state[idx]*self.J[idx, neighbors] @ state[neighbors].T + 2*h*state[idx]
            #print(delta_energy)
            #print(alpha, torch.exp(-self.beta*delta_energy))
            if delta_energy <= 0 or alpha<torch.exp(-self.beta*delta_energy):
                state[idx] = -state[idx]
            else:
                None
            #print(state)
            if i+1 >s_step and (i+1)%interval ==0:
                s.append(state.tolist())
        s = torch.tensor(s)
        #print(s)
        return s

            

    def get_bonds(self):
        G = self.graph
        neighbors = []
        for i in range(self.n):
            a = list(G.neighbors(i))
            a.sort()
            neighbors.append(a)
        edges = []
        for i in range(len(neighbors)):
            for node in neighbors[i]:
                if node > i:
                    edges.append([i, node])
        nodes_bonds = []
        for i in range(self.n):
            nodes_bonds.append([])
            for j in range(len(edges)):
                bond = edges[j]
                if i in bond:
                    nodes_bonds[i].append(edges.index(bond))
        '''neighbors of physical site'''
        self.neighbors = neighbors 
        '''sites of edges'''
        self.edges = edges
        '''edges of sites'''
        self.nodes_bonds = nodes_bonds

    def copy_tensor(self, n_idx, d=2):
        if n_idx == 1:
            copy_tensor = torch.ones(d, device=self.device, dtype=self.dtype)
        else:
            copy_tensor = torch.zeros(d**n_idx, device=self.device, dtype=self.dtype)
            copy_tensor[0] = copy_tensor[-1] = 1
            copy_tensor = copy_tensor.reshape([d]*n_idx)
        return copy_tensor
    
    def generate_tensors(self, hyper=False):
        n = self.n
        J = self.J.to(self.dtype)
        beta = self.beta
        edges = self.edges 
        
        if hyper==True:
            
            BMs = []
            for edge in edges:
                i, j = edge
                a = torch.exp(beta*J[i, j])
                b = torch.exp(-beta*J[i, j])
                B = torch.tensor([[a, b], [b, a]], dtype=self.dtype, device=self.device)
                BMs.append(B)
            return BMs
        elif hyper==False :
            if self.coupling == 'fe':
                e = 'abcdefghijklmnopqrstuvwxyz'
                nodes_bonds = self.nodes_bonds
                Bs = []
                for edge in edges:
                    i, j = edge
                    a = torch.sqrt(torch.cosh(beta*J[i, j])/2)
                    b = torch.sqrt(torch.sinh(beta*J[i, j])/2)
                    B = torch.tensor([[a+b, a-b], [a-b, a+b]], dtype=self.dtype, device=self.device)
                    Bs.append(B)
                tensors = []
                for j in range(n):
                    bonds = nodes_bonds[j]
                    degree  = len(bonds)
                    t = self.copy_tensor(degree)
                    for k in range(degree):
                        bond = bonds[k]
                        
                        eq = e[:degree] + ','+ e[k] + 'A' + '->' + e[:degree].replace(e[k], 'A')
                        #print(eq)
                        t = oe.contract(eq, *[t, Bs[bond]])
                    tensors.append(t)
            else:
                A = 'abcdefghijklmnopqrstuvwxyz'
                nodes_bonds = self.nodes_bonds
                edges = self.edges
                BMs = []
                for edge in edges:
                    i, j = edge
                    a = torch.exp(beta*J[i, j])
                    b = torch.exp(-beta*J[i, j])
                    B = torch.tensor([[a, b], [b, a]], dtype=self.dtype, device=self.device)
                    Q, R = torch.linalg.qr(B)
                    BMs.append([Q, R])
                tensors = []
                for j in range(n):
                    bonds = nodes_bonds[j]
                    degree  = len(bonds)
                    t = self.copy_tensor(degree)
                    for k in range(degree):
                        e = bonds[k]
                        dim = edges[e].index(j)
                        qr = BMs[e][dim]
                        if dim ==0 :
                            eq = A[:degree] + ','+ A[k] + 'A' + '->' + A[:degree].replace(A[k], 'A')
                        elif dim ==1 :
                            eq = A[:degree] + ','+ 'A' + A[k] + '->' + A[:degree].replace(A[k], 'A')
                        t = oe.contract(eq, *[t, qr])
                    tensors.append(t)
            return tensors          


if __name__ == '__main__':
    from os.path import abspath, dirname
    # L = 5
    # dim = 2
    # h=0.01
    # beta_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1, 2]
    # for i in range(len(beta_list)):
    #     Is = Ising(L=L, dim=dim, beta=beta_list[i], dtype=torch.float32)
    #     samples = Is.MH(10000, h=h)
    #     s = samples[:, ::2]
    #     print(s.size())
    #     print(torch.mean(samples, dim=0))
    #     print(abspath(dirname(__file__)).strip('module')+'/database/ising/'+'L{}_dim{}_beta{}_h{}.pt'.format(L, dim, beta_list[i], h))
    #     torch.save((samples, s), abspath(dirname(__file__)).strip('module')+'/database/ising/'+'L{}_dim{}_beta{}_h{}.pt'.format(L, dim, beta_list[i], h))
    # #print(Is.neighbors)
    #     #print(abspath(dirname(__file__)).strip('Btensors')+'qec/database/ising/'+'L{}_dim{}_beta{}_h{}.pt'.format(L, dim, beta_list[i], h))
    n=25
    degree=4
    h=0.3
    beta_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1, 2]
    Is = RandomIsing(n=n, degree=degree, dtype=torch.float32, coupling='uni')
    for i in range(len(beta_list)):
       
        samples = Is.MH(10000, beta=beta_list[i], h=h)
        s = samples
        #print(s.size())
        print(torch.mean(samples, dim=0))
        #print(samples)
        torch.save((samples, s), abspath(dirname(__file__)).strip('module')+'/database/ising/'+'n{}_degree{}_beta{}_h{}.pt'.format(n, degree, beta_list[i], h))