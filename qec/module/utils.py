import torch
import numpy as np
import networkx as nx
import opt_einsum as oe
from math import floor
from copy import deepcopy
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('module'))
from module.mod2 import mod2
from os.path import abspath, dirname


mod2 = mod2()


def generate_graph(n, m, degree=3, seed=0,  G_type='rrg'):
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
    return G

'''read code which is generated by code_generator.py'''
def read_code(d, k, seed, c_type='sur', n=None):
    if c_type == 'qcc':
        return torch.load(abspath(dirname(__file__)).strip('module')+'code/'+c_type+'_n{}_k{}'.format(n, k))
    else:
        return torch.load(abspath(dirname(__file__)).strip('module')+'code/'+c_type+'_d{}_k{}_seed{}'.format(d, k, seed))

def read_data(d, k, seed, c_type='sur'):
    return torch.load(abspath(dirname(__file__)).strip('module')+'database/'+c_type+'_d{}_k{}_seed{}.pt'.format(d, k, seed))


def btype(n, l):
    '''turn a normal number n to a binary number in torch.tensor type with length l'''
    b = torch.zeros(l)
    for i in range(l-1, -1, -1):
        b[i] = (n//2**i)%2
        if n > 2**i:    
            n -= 2**i
    return b


def bbtype(n, l):
    '''turn batch normal numbers with size (batch, 1) to binary numbers in torch.tensor type with size (batch, l)'''
    b = torch.zeros(n.size(0), l)
    for i in range(l-1, -1, -1):
        b[:, i] = (n//2**i)%2
        if n[i] > 2**i:    
            n[i] -= 2**i
    return b
        

def exact_config(D, N, device='cpu', dtype=torch.float64):
    '''generate (N, D) exact configurations where N <= 2**D'''
    config = torch.empty((N, D), device=device, dtype=dtype)
    for i in range(N - 1, -1, -1):
        num = i
        for j in range(D - 1, -1, -1):
            config[i, D - j - 1] = num // 2 ** j
            if num - 2 ** j >= 0:
                num -= 2 ** j
    return config 

def PCM_to_Stabilizer(PCM):
    '''turn parity check matrix(PCM) to stabilizer rep'''
    n = int(PCM.size(1)/2)
    M = torch.zeros_like(PCM)
    M[:, :n], M[:, n:] =  PCM[:, n:], PCM[:, :n]
    return mod2.xyz(M)
    

def PCM(g_stabilizer):
    '''get parity check matrix from stabilizers'''
    n = g_stabilizer.size(1)
    M = mod2.rep(g_stabilizer)
    PCM = torch.zeros_like(M)
    PCM[:, :n], PCM[:, n:] =  M[:, n:], M[:, :n]
    return PCM

def Hx_Hz(g_stabilizer):
    '''get PCM Hx and Hz'''
    g = g_stabilizer
    hx, hz = [], []
    for i in range(g.size(0)):
        if (g[i]%2).sum() != 0:
            hx.append(g[i])
        else:
            hz.append(g[i])
    Hx = mod2.rep(torch.vstack(hx))
    Hz = mod2.rep(torch.vstack(hz))
    return Hx[:, :g.size(1)], Hz[:, g.size(1):]


def error_solver(PCM, b):
    '''find the errors satisfy the E @ PCM.T = I'''
    #n = int(PCM.size(1)/2)
    errors = mod2.solve(PCM, b)
    #errors1 = torch.zeros_like(errors)
    #errors1[:, :n], errors1[:, n:] = errors[:, n:], errors[:, :n]
    return mod2.xyz(errors)
    

class Errormodel():
    def __init__(self, e_rate = 0.2, e_model='depolarized'):
        '''
            error model class
            e_rate: physical error rate
            e_model：error model
        '''
        self.e_rate = e_rate
        if e_model == 'depolarized':
            self.single_p = np.array([1-e_rate, e_rate/3, e_rate/3, e_rate/3])
        elif e_model == 'x_model':
            self.single_p = np.array([1-e_rate, e_rate-2e-9, 1e-9, 1e-9])

    def generate_error(self, n, m=1, seed=0):
        '''
        generate errors from given error model with size (m, n)
        n: number of physical qubits
        m: number of errors
        '''
        single_p = self.single_p
        if seed != False:
            np.random.seed(seed)
        if m==1:
            error = torch.tensor(np.random.choice([0,1,2,3], [n], p=single_p))
        else:
            error = torch.tensor(np.random.choice([0,1,2,3], [m, n], p=single_p))
        return error

    def pure(self, pure_es, syndrome, device='cuda:0', dtype=torch.float64):
        '''
            get pure error with syndrome
            pure_es: pure_errors of a definite code with size (code.m, code.n)
            syndrome: input syndrome or syndromes with size (batch, code.m)
        '''
        s = syndrome.to(device=device, dtype=dtype)
        mod2.to(device=device,dtype=dtype)
        return mod2.confs_to_opt(s, pure_es)

    def log_probability(self, opts, device, dtype):
        '''get log probabilities of input operators'''
        opts = opts.to(int)
        p = torch.tensor(self.single_p, dtype=dtype, device=device)
        prob = p[opts]
        logp = torch.log(prob).sum(dim=1)
        return logp
    
    
    def configs(self, sta, log, pe, opts, seq='mid'):
        '''
            get ELS configurations of input operators
            opts: in put operators
            sta: stabilizers of code
            log: logicals of code
            pe: pure errors of code
        '''
        syndrome = mod2.commute(opts, sta)
        sta_conf = mod2.commute(opts, pe)
        log_conf = mod2.commute(opts, log)
        
        k = int(log.size(0)/2)
        index = []
        for i in range(k):
            index.append(2*i+1)
            index.append(2*i)
        log_conf = log_conf[:,index]
        
        if seq == 'mid':
            config = torch.hstack([syndrome, log_conf, sta_conf])
        elif seq == 'fin':
            config = torch.hstack([syndrome, sta_conf, log_conf])
        return config
    # def ising_error(self, n_s, L=5, dim=2, beta=0.3, h=1):
    #     _, s = torch.load(abspath(dirname(__file__)).strip('module')+'database/ising/'+'L{}_dim{}_beta{}_h{}.pt'.format(L, dim, beta, h))
    #     s = (1 - s[:n_s, :])/2
    #     s = s.view(1, -1).squeeze()
    #     indices = torch.nonzero(s).squeeze()
    #     s[indices] = torch.tensor(np.random.choice([1, 2, 3], [len(indices)], p=[1/3, 1/3, 1/3]), dtype=s.dtype)
    #     s = s.view(n_s, -1)
    #     return s

    def ising_error(self, n_s, n=13, degree=4, beta=0.3, h=1.0):
        _, s = torch.load(abspath(dirname(__file__)).strip('module')+'database/ising/'+'n{}_degree{}_beta{}_h{}.pt'.format(n, degree, beta, h))
        #print(s)
        a = s.sum(0)
        #print(s.size())
        s = (1 - s[:n_s, :])/2
        s = s.view(1, -1).squeeze()
        indices = torch.nonzero(s).squeeze()
        s[indices] = torch.tensor(np.random.choice([1, 2, 3], [len(indices)], p=[1/3, 1/3, 1/3]), dtype=s.dtype)
        s = s.view(n_s, -1)
        return s, a
#E = Errormodel()
#E.ising_error(n_s=, beta=0.7)

def batch_eq(eq, link_info, n_l=10000):
    a = ','
    c = eq[-1]
    if c != '>':
        eq = eq[:-1]
    b = ' -> '
    x = oe.get_symbol(n_l)
    eq1 = eq.strip(b)
    list_eq = eq1.split(a)
    for i in range(len(link_info)):
        ct = link_info[i][1]
        if list_eq[ct].count(x) == 0:
            list_eq[ct] += x
    eq1 = a.join(list_eq)
    eq1 += ' -> '
    if c != '>':
        eq1 += c
    eq1 += x
    return eq1


class CodeTN():
    def __init__(self, Code, multi=False, device='cuda:0', dtype=torch.float32):
        
        self.device = device
        self.dtype = dtype
        
        self.multi = multi
        self.n = Code.n
        self.m = Code.m
        self.g_stabilizer = Code.g_stabilizer
        self.pure_es = Code.pure_es
        self.logical_opt = Code.logical_opt
        self.physical_qubits = Code.physical_qubits
 
        self.neighbors = [Code.physical_qubits[i] for i in range(len(Code.physical_qubits))]
        self.shapes = self.shapes()

        mod2.to(device, dtype)
    
    def probability(self, error, single_p, confs):
        batch = confs.size(0)
        stabilizers = mod2.confs_to_opt(confs, self.g_stabilizer)
        batch_e = torch.vstack([error]*batch)
        new_e = mod2.opt_prod(batch_e, stabilizers).to(int)
        p = torch.tensor(single_p, dtype=self.dtype, device=self.device)
        prob = p[new_e] 
        P = torch.prod(prob, dim=1)
        return P
    

    def multi_error_4c(self, error):
        L_opt = self.logical_opt
        if len(error.size()) == 1:
            b_e = torch.tensor([error.tolist()]*4, device=self.device)
            return mod2.opt_prod(b_e, L_opt)
        else:
            multi = error.size(0)
            batch_multi_e = []
            for j in range(multi):
                batch_multi_e.append(torch.tensor([error[j].tolist()]*4, device=self.device))
            batch_multi_e = torch.cat(batch_multi_e, dim=0)
            m_L_opt = torch.cat([L_opt]*multi, dim=0)
            return mod2.opt_prod(batch_multi_e, m_L_opt)
    
    def generate_tensors(self, error, single_p, norm=False, alpha=1):
        phy_q = self.physical_qubits
        tensors = []
        if self.multi== False:
            for i in range(len(phy_q)):
                gs = self.g_stabilizer[phy_q[i]]
                l = len(gs)
                shape = [2]*l
                conf = exact_config(l, 2**l, device=self.device, dtype=self.dtype)
                opt = mod2.confs_to_opt(conf, gs[:, i].unsqueeze(1)).unsqueeze(1)
                #print(error[[i]*(2**l)].unsqueeze(0).T)
                batch_e = error[[i]*(2**l)].unsqueeze(0).T
                #batch_e = torch.cat([error[i]]*(2**l)).unsqueeze(1)
                new_e = mod2.opt_prod(batch_e, opt).to(int)
                t = alpha*torch.tensor(single_p, device=self.device, dtype=self.dtype)[new_e].reshape(shape=shape)
                tensors.append(t)
        else:
            multi = error.size(0)
            for i in range(len(phy_q)):
                gs = self.g_stabilizer[phy_q[i]]
                l = len(gs)
                shape = [multi]+[2]*l
                conf = exact_config(l, 2**l, device=self.device, dtype=self.dtype)
                opt = mod2.confs_to_opt(conf, gs[:, i].unsqueeze(1))
                multi_opt = torch.tensor([opt.tolist()]*multi, device=self.device).view([multi*(2**l), -1])

                batch_multi_e = torch.cat([error[:, i].unsqueeze(1)]*2**l, dim=1).reshape(multi*2**l, -1)
                new_e = mod2.opt_prod(batch_multi_e, multi_opt).to(int)
                t = alpha*torch.tensor(single_p, device=self.device, dtype=self.dtype)[new_e].reshape(shape=shape)
                tensors.append(t)
        log_norm = - len(phy_q)*torch.log(torch.tensor(alpha, device=self.device))
        if norm == True:
            
            for i in range(self.n):#floor(self.n*4/5)
                log_norm += torch.log(torch.norm(tensors[i]))
                tensors[i] = tensors[i]/torch.norm(tensors[i])
        return tensors, log_norm
        #return None

    def shapes(self):
        neighbors = self.neighbors
        shapes = []
        for i in range(len(neighbors)):
            if self.multi == False:
                shape = []
            else:
                shape = [4]
            for j in neighbors[i]:
                shape.append(2)
            shapes.append(shape)
        return shapes

    def ein_eq(self):
        neighbors = self.neighbors
        x = oe.get_symbol(10001)
        eq = ''
        for neigh in neighbors:
            if self.multi == False:
                label = ''
            else:
                label = x
            for j in neigh:
                symble = oe.get_symbol(j)
                label += symble
            eq += label
            eq += ','
        eq = eq[:-1] + ' -> ' 
        if self.multi == True:
            eq += x
        return eq
    
    def slicing_node(self, frozen_nodes):
        link_info = []
        neighbors = deepcopy(self.neighbors)
        for node in frozen_nodes:
            idx = frozen_nodes.index(node)
            for edge in range(len(neighbors)):
                if node in neighbors[edge]:
                    dim = neighbors[edge].index(node)
                    neighbors[edge].remove(node)
                    link_info.append([idx, edge, dim])
        self.neighbors = neighbors
        return link_info


    def new_tensors(self, link_info, configure, tensors):
        
        w = torch.zeros(configure.size(0), 2, device=self.device, dtype=self.dtype)
        w[:, 0] = (configure+1)%2
        w[:, 1] = configure

        tensors = deepcopy(tensors)
 
        for i in range(len(link_info)):
            idx = link_info[i][0]
            ct = link_info[i][1]
            dim = link_info[i][2]
  
            tensors[ct] = torch.tensordot(tensors[ct], w[idx], dims=([dim], [0]))
            
        return tensors
    
    def new_tensors_batch(self, link_info, configures, tensors):#, device='cuda:0', dtype=torch.float64):
        s = 'abcdefghijklmnopqrstuvwxyz'
        w = torch.zeros(configures.size(0), configures.size(1), 2, device=self.device, dtype=self.dtype)
        w[:, :, 0] = (configures+1)%2
        w[:, :, 1] = configures

        tensors = deepcopy(tensors)
        shapes = deepcopy(self.shapes)

        for i in range(len(link_info)):
            
            idx = link_info[i][0]
            ct = link_info[i][1]
            dim = link_info[i][2]
            if self.multi == True:
                dim += 1
            shape = shapes[ct]
            if tensors[ct].size(-1) == 2:
                shape[-1] = configures.size(0)
                tensors[ct] = torch.tensordot(tensors[ct], w[:, idx, :], dims=([dim], [1])).view(shape)
            else:
                eq = s[:len(shape)]
                eq1 = ','
                eq1 += eq[-1]
                eq1 += eq[dim]
                eq1 = eq + eq1
                eq = eq.replace(eq[dim], '')
                eq1 += '->'
                eq1 += eq
                #print(tensors[ct].size())
                tensors[ct] = torch.einsum(eq1, tensors[ct], w[:, idx, :])
                shape.remove(2)
                #print(eq1)
                #print(tensors[ct].size())
        return tensors
    
class SurfacecodeTN():
    def __init__(self, code, dtype=torch.float64, device='cpu'):
        self.L = code.d + code.d-1
        self.device, self.dtype = device, dtype
        self.n = code.n
        self.m = code.m
        self.g_stabilizer = code.g_stabilizer
        self.pure_es = code.pure_es
        self.logical_opt = code.logical_opt
        self.physical_qubits = code.physical_qubits

        self.graph = code.G
        self.get_bonds()

    def get_bonds(self):
        G = self.graph
        neighbors = []
        for i in range(self.L**2):
            a = list(G.neighbors(i))
            a.sort()
            neighbors.append(a)
        edges = []
        for i in range(len(neighbors)):
            for node in neighbors[i]:
                if node > i:
                    edges.append([i, node])
        nodes_bonds = []
        for i in range(self.L**2):
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
        copy_tensor = torch.zeros(d**n_idx, device=self.device, dtype=self.dtype)
        copy_tensor[0] = copy_tensor[-1] = 1
        copy_tensor = copy_tensor.reshape([d]*n_idx)
        return copy_tensor    

    def generate_tensors(self, error, single_p, multi=False):
        if multi == False:
            None
        elif multi == True:
            bdim = error.size(0)
            tensors = []
            for i in range(self.L**2):
                if i%2 == 1:
                    degree = len(self.nodes_bonds[i])
                    copy = self.copy_tensor(degree).unsqueeze(0)
                    copies = torch.cat([copy]*bdim, dim=0)
                    tensors.append(copies)
                else:
                    phy_q = self.physical_qubits
                    a = int(i/2)
                    gs = self.g_stabilizer[phy_q[a]]
                    l = len(gs)
                    shape = [bdim]+[2]*l
                    conf = exact_config(l, 2**l, device=self.device, dtype=self.dtype)
                    opt = mod2.confs_to_opt(conf, gs[:, a].unsqueeze(1))
                    multi_opt = torch.cat([opt]*bdim, dim=0).view([bdim*(2**l), -1])#torch.tensor([opt.tolist()]*bdim, device=self.device).view([bdim*(2**l), -1])
                    batch_multi_e = torch.cat([error[:, a].unsqueeze(1)]*2**l, dim=1).reshape(bdim*2**l, -1)
                    new_e = mod2.opt_prod(batch_multi_e, multi_opt).to(int)
                    t = torch.tensor(single_p, device=self.device, dtype=self.dtype)[new_e].reshape(shape=shape)
                    tensors.append(t)
        return tensors

if __name__ == '__main__':
    None
    
