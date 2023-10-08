
import torch
import numpy as np
from numpy.linalg import matrix_power
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('module'))
from module.utils import generate_graph, PCM, PCM_to_Stabilizer, error_solver, Hx_Hz
from module.mod2 import mod2
mod2 = mod2()

class Loading_code():
    def __init__(self, info):
        self.g_stabilizer, self.logical_opt, self.pure_es = info[0], info[1], info[2]
        self.n, self.m = self.g_stabilizer.size(1), self.g_stabilizer.size(0)
        self.physical_qubits = self.get_physical_qubits()
        self.PCM = PCM(self.g_stabilizer)
    def get_physical_qubits(self):
        gs = self.g_stabilizer
        n, m = self.n, self.m
        phys = {}.fromkeys(range(n))
        for i in range(n):
            phys[i] = []
            for j in range(m):
                if gs[j, i] != 0:
                    phys[i].append(j)
        return phys


class Abstractcode():
    '''
        class of general code
        input the stabilizers or PCM will generte the complete code.
    '''
    def __init__(self, g_stabilizer, intype='xyz', complete=True):
        if intype == 'xyz':
            self.g_stabilizer = g_stabilizer
            self.PCM = PCM(self.g_stabilizer)
        elif intype == 'bin':
            self.g_stabilizer = PCM_to_Stabilizer(g_stabilizer)
            self.PCM = g_stabilizer

        self.n, self.m = self.g_stabilizer.size(1), self.g_stabilizer.size(0)

        
        self.physical_qubits = self.get_physical_qubits()
        if complete == True:
            self.logical_opt = self.get_logical_opt()
            self.pure_es = self.pure_errors()
        elif complete == False:
            self.logical_opt = None
            self.pure_es = None



    def get_physical_qubits(self):
        gs = self.g_stabilizer
        n, m = self.n, self.m
        phys = {}.fromkeys(range(n))
        for i in range(n):
            phys[i] = []
            for j in range(m):
                if gs[j, i] != 0:
                    phys[i].append(j)
        return phys
    
    def get_logical_opt(self):
        Hx, Hz = Hx_Hz(self.g_stabilizer)
        x = mod2.kernel(Hz)
        z = mod2.kernel(Hx)*2
        lx_lz = mod2.Schmidt(torch.vstack([x, z]))
        L =torch.vstack(lx_lz)
        return L


    def pure_errors(self):
        k = self.n-self.m
        b = torch.eye(self.m)
        errors = error_solver(self.PCM, b)
        #print(errors)
        if mod2.commute(errors, self.logical_opt).sum() == 0:
            return errors
        else:
            for i in range(0, 2*k, 2):
                L = self.logical_opt[[i, i+1]]
                com = mod2.commute(errors, L)
                row = [i for i in range(self.m) if com[i].sum() == 1]
                idx = (com[row]-1).nonzero()[:, 1]
                errors[row] = mod2.opt_prod(errors[row], L[[idx]])
                row1 = [i for i in range(self.m) if com[i].sum() == 2]
                errors[row1] = mod2.opt_prod(errors[row1], L[0])
                errors[row1] = mod2.opt_prod(errors[row1], L[1])
        for i in range(self.m):
            conf = mod2.commute(errors[i], errors)
            idx = conf.nonzero().squeeze()
            #print(idx)
            sta = self.g_stabilizer[idx]
            errors[i] = mod2.opts_prod(torch.vstack([errors[i], sta]))
        return errors    


class Surfacecode():
    def __init__(self, d):
        self.d = d
        self.n = d**2 + (d-1)**2
        self.m = 2*d*(d-1)
        self.G = generate_graph(2*d-1, 2*d-1, G_type='2D')

        self.g_stabilizer = self.get_generators_of_stabilizers()
        self.physical_qubits = self.get_physical_qubits()
        self.logical_opt = self.get_logical_opt()
        self.PCM = PCM(self.g_stabilizer)

        self.pure_es = self.pure_errors()

    def get_generators_of_stabilizers(self):
        s = torch.zeros(self.m, self.n)
        for l in range(self.m):
            label = 2*l+1
            n_row = label//(2*self.d-1)
            if n_row % 2 == 0:
                for j in list(self.G.neighbors(label)):
                    s[l, int(j/2)] = 2
            else:
                for j in list(self.G.neighbors(label)):
                    s[l, int(j/2)] = 1
        return s
    
    def get_physical_qubits(self):
        node_label = [i for i in range(self.n)]
        physical_qubits = {}.fromkeys(node_label)
        for i in range(self.n):
            link_s = np.array(list(self.G.neighbors(2*i)))
            physical_qubits[i] = list(((link_s-1)/2).astype(int))
        return physical_qubits
    
    def get_logical_opt(self):
        n = self.n
        L_opt = torch.zeros(4, n)
        for i in range(self.d):
            L_opt[1, i] = 1
            L_opt[2, i*(2*self.d-1)] = 2
        L_opt[3] = mod2.opt_prod(L_opt[1], L_opt[2])

        return L_opt

    def pure_errors(self):
        b = torch.eye(self.m)
        errors = error_solver(self.PCM, b)
        commutation = mod2.commute(errors, self.logical_opt[1:])
        row = commutation.sum(1).nonzero().squeeze()
        idx = (commutation[row]-1).nonzero()[:, 1]
        errors[row] = mod2.opt_prod(errors[row], self.logical_opt[idx+1])
        for i in range(self.m):
            conf = mod2.commute(errors[i], errors)
            idx = conf.nonzero().squeeze()
            #print(idx)
            sta = self.g_stabilizer[idx]
            errors[i] = mod2.opts_prod(torch.vstack([errors[i], sta]))

        return errors     

class Rotated_Surfacecode():
    def __init__(self, d):
        self.d = d
        self.n = d**2
        self.m = d**2 - 1
        self.g_stabilizer = self.get_generators_of_stabilizers()
        self.physical_qubits = self.get_physical_qubits()
        self.logical_opt = self.get_logical_opt()
        self.PCM = PCM(self.g_stabilizer)
        self.pure_es = self.pure_errors()

    def get_generators_of_stabilizers(self):
        d = self.d
        s = torch.zeros(self.m, self.n)
        for i in range(d-1):
            for j in range(d-1):
                idx = i*(d-1)+j
                if (i+j)%2==0:
                    s[idx, [i*d+j, i*d+j+1, (i+1)*d+j, (i+1)*d+j+1]] = 1
                else:
                    s[idx, [i*d+j, i*d+j+1, (i+1)*d+j, (i+1)*d+j+1]] = 2
        a = (d-1)**2
        b = int((d-1)/2)
        for i in range(b):
            s[a+i, [2*i+1, 2*i+2]] = 1
            #print(2*i+1, 2*i+2)
        for i in range(b):
            s[a+b+i, [d*(d-1)+2*i, d*(d-1)+2*i+1]] = 1
            #print(d*(d-1)+2*i, d*(d-1)+2*i+1)
        for i in range(b):
            s[a+2*b+i, [d*(2*i+1)+d-1, d*(2*i+2)+d-1]] = 2
            #print(d*(2*i+1)+d-1, d*(2*i+2)+d-1)
        for i in range(b):
            s[a+3*b+i, [2*d*i, 2*d*i+d]] = 2
            #print(2*d*i, 2*d*i+d)
        return s

    def get_physical_qubits(self):
        gs = self.g_stabilizer
        n, m = self.n, self.m
        phys = {}.fromkeys(range(n))
        for i in range(n):
            phys[i] = []
            for j in range(m):
                if gs[j, i] != 0:
                    phys[i].append(j)
        return phys 

    def get_logical_opt(self):
        n = self.n
        L_opt = torch.zeros(2, n)
        for i in range(self.d):
            L_opt[1, i] = 2
            L_opt[0, i*self.d] = 1
        return L_opt

    def pure_errors(self):
        k = self.n-self.m
        b = torch.eye(self.m)
        errors = error_solver(self.PCM, b)
        #print(errors)
        if mod2.commute(errors, self.logical_opt).sum() == 0:
            return errors
        else:
            for i in range(0, 2*k, 2):
                L = self.logical_opt[[i, i+1]]
                com = mod2.commute(errors, L)
                row = [i for i in range(self.m) if com[i].sum() == 1]
                idx = (com[row]-1).nonzero()[:, 1]
                errors[row] = mod2.opt_prod(errors[row], L[[idx]])
                row1 = [i for i in range(self.m) if com[i].sum() == 2]
                errors[row1] = mod2.opt_prod(errors[row1], L[0])
                errors[row1] = mod2.opt_prod(errors[row1], L[1])
            
        for i in range(self.m):
            conf = mod2.commute(errors[i], errors)
            idx = conf.nonzero().squeeze()
            #print(idx)
            sta = self.g_stabilizer[idx]
            errors[i] = mod2.opts_prod(torch.vstack([errors[i], sta]))
        return errors 


def s_matrix(dim):
    matrix = np.eye(dim)
    matrix = np.concatenate((matrix[:, -1:], matrix[:, :-1]), axis=1)
    return matrix

def x_matrix(l, m):
    return np.kron(s_matrix(l), np.eye(m))

def y_matrix(l, m):
    return np.kron(np.eye(l), s_matrix(m))

class QuasiCyclicCode:
    def __init__(self, l, m, polynomial_a, polynomial_b) -> None:
        assert len(polynomial_a) == len(polynomial_b) == 3
        self.a_matrices = [
            matrix_power(m_function(l, m), poly) % 2
            for m_function, poly in zip([x_matrix, y_matrix, y_matrix], polynomial_a)
        ]
        self.a_matrix = sum(self.a_matrices) % 2
        self.b_matrices = [
            matrix_power(m_function(l, m), poly) % 2
            for m_function, poly in zip([y_matrix, x_matrix, x_matrix], polynomial_b)
        ]
        self.b_matrix = sum(self.b_matrices) % 2
        self.hx = np.concatenate([self.a_matrix, self.b_matrix], axis=1)
        self.hz = np.concatenate([self.b_matrix.T, self.a_matrix.T], axis=1)
        hx, hz = torch.from_numpy(self.hx), torch.from_numpy(self.hz)

        a = mod2.indep(hx)
        b = mod2.indep(hz)
        c = torch.zeros(a.size(0), b.size(1))
        d = torch.zeros(b.size(0), a.size(1))
        self.PCM = torch.vstack([torch.hstack([a, c]), torch.hstack([d, b])]).long()
        self.stabilizers = mod2.xyz(self.PCM)

class Toric():
    def __init__(self, d):
        self.d = d
        self.n = 2*d**2
        self.m = 2*d**2 - 2

        g = self.get_generators_of_stabilizers()
        self.PCM = mod2.L_indep(PCM(g))
        self.g_stabilizer = PCM_to_Stabilizer(self.PCM)

    def get_generators_of_stabilizers(self):
        d = self.d
        #m1 = int(self.m/2)
        s = torch.zeros(d, d, self.n)
        for i in range(d):
            for j in range(d):
                if j == d-1:
                    s[i, j][[i*2*d+j, ((2*i+2)*d+j)%self.n, (i*2+1)*d+j, i*2*d+j+1]]=2
                    #print(i*2*d+j, ((2*i+2)*d+j)%self.n, (i*2+1)*d+j, i*2*d+j+1)
                else:
                    s[i, j][[i*2*d+j, ((2*i+2)*d+j)%self.n, (i*2+1)*d+j, (i*2+1)*d+j+1]]=2
                    #print(i*2*d+j, ((2*i+2)*d+j)%self.n, (i*2+1)*d+j, (i*2+1)*d+j+1)
        s = s.reshape(-1, self.n)
        s1 = torch.zeros(d, d, self.n)
        for i in range(d):
            for j in range(d):
                if i == 0 and j!=0:
                    s1[i, j][[i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, self.n-d+j]]=1
                    # print(i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, self.n-d+j)
                elif i!=0 and j == 0:
                    s1[i, j][[i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, (i*2-1)*d+j]]=1
                    # print(i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, (i*2-1)*d+j)
                elif i ==0 and j ==0:
                    s1[i, j][[i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, self.n-d+j]]=1
                    # print(i*2*d+j, (i*2+1)*d+j, (i*2+1)*d-1, self.n-d+j)
                else:
                    s1[i, j][[i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, (i*2-1)*d+j]]=1
                    # print(i*2*d+j, (i*2+1)*d+j, i*2*d+j-1, (i*2-1)*d+j)
        s1 = s1.reshape(-1, self.n)
        return torch.vstack((s[1:], s1[:-1]))

class Sur_3D():
    def __init__(self, d):
        self.d = d
        info = torch.load(abspath(dirname(__file__))+'/3d_sc/3d_sc_d{}.pt'.format(d))
        
        g = torch.cat([info[0], info[1]*2], dim=0)
        self.PCM = mod2.L_indep(PCM(g))
        self.n = g.size(1)
        self.m = self.n-1
        self.g_stabilizer = PCM_to_Stabilizer(self.PCM)
        
        
        self.logical_opt = mod2.xyz(info[2].reshape(4, 2*self.n))
        self.PCM = PCM(self.g_stabilizer)
        self.physical_qubits = self.get_physical_qubits()

        #self.pure_es = pure_errors(self.PCM, self.logical_opt)
        

    def get_physical_qubits(self):
        n, m = self.n, self.m
        gs = self.g_stabilizer
        phys = {}.fromkeys(range(n))
        for i in range(n):
            phys[i] = []
            for j in range(m):
                if gs[j, i] != 0:
                    phys[i].append(j)
        return phys
  
     
    

class Random_Srting_code():
    def __init__(self, nv, degree, seed=0):
        self.nv = nv
        self.ne = int(nv*degree/2)
        self.G = generate_graph(n=nv, m=1, degree=degree, seed=seed, G_type='rrg')
        self.n = self.nv+5*self.ne
        self.m = self.ne*4

        self.get_stabilizer()
    
    def basic_sting(self):
        bs = torch.tensor([[1, 1, 0, 0, 0],
                           [2, 2, 2, 0, 0],
                           [0, 0, 2, 2, 2],
                           [0, 0, 0, 1, 1]])
        return bs
    
    def get_stabilizer(self):
        bs = self.basic_sting()
        g = torch.zeros(self.m, self.n)
        print(g.size())
        edges = list(self.G.edges())
        for i in range(len(edges)):
            j, k = edges[i]
            if i <= self.nv-1:
                #print(i, 'a', i*6+1,i*6+6-1)
                g[i*4:i*4+4, i*6+1:i*6+6] = bs
            else:
                #print(i, self.nv*6+(i-self.nv)*5,self.nv*6+(i-self.nv)*5+5)
                g[i*4:i*4+4, self.nv*6+(i-self.nv)*5:self.nv*6+(i-self.nv)*5+5] = bs
            #print(i, g)
            g[i*4, j*6] = 1
            g[i*4+3, k*6] = 1
        self.g_stabilizers = g
    
if __name__ == '__main__':
    from module.utils import PCM, Errormodel, Hx_Hz, exact_config
    from copy import deepcopy
    import numpy as np
    S = Toric(3)
    a = S.PCM
    print(a.size(0)) 
    b = mod2.rank(a)
    print(b)
    c = mod2.L_indep(a)
    print(c.size(0))
    # d=7
    # T = Toric(d=d)#Sur_3D(d=d)#
    # code = Abstractcode(T.g_stabilizer)
    # # print(code.pure_es.size())
    # print(mod2.commute(code.g_stabilizer, code.g_stabilizer).sum())
    # print((mod2.commute(code.g_stabilizer, code.pure_es)-torch.eye(code.m)).sum().item())
    # print(mod2.commute(code.g_stabilizer, code.logical_opt).sum().item())
    # print(mod2.commute(code.pure_es, code.logical_opt).sum().item())
    # print(mod2.commute(code.logical_opt, code.logical_opt))

    # b = torch.ones(T.m)
    # PCM = PCM(T.g_stabilizer)
    # print(mod2.rank(PCM))
    # #solution = mod2.solve(PCM, b)
    # r, b1 = mod2.row_echelon(PCM, b)
    # solution = mod2.solve(r, b1)
    # print((r@solution.T)%2-b1)



