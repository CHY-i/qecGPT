import torch
from copy import deepcopy
#from utils import exact_config

def row_sum_sort(t):
    x = []
    for i in range(t.size(0)):
        x.append([t[i].sum().item(), i])
    x.sort()
    x = torch.tensor(x)
    return t[x[:, 1]]

def exact_config(D, N, device='cpu', dtype=torch.float64):
    config = torch.empty((N, D), device=device, dtype=dtype)
    for i in range(N - 1, -1, -1):
        num = i
        for j in range(D - 1, -1, -1):
            config[i, D - j - 1] = num // 2 ** j
            if num - 2 ** j >= 0:
                num -= 2 ** j
    return config 

class mod2():
    def __init__(self, dtype=int, device='cpu') -> None:
        self.dtype=dtype
        self.device=device

    def to(self, device, dtype):
        self.dtype = dtype
        self.device = device
        
    def rep(self, input):
        '''turn [0, 1, 2, 3] to binary [[0, 0], [0, 1], [1, 0], [1, 1]] '''
        if len(input.size()) <=1:
            input = torch.unsqueeze(input, dim=0)
        n, m = input.size(1), input.size(0)
        bin = torch.zeros(m, n*2, device=self.device)
        #print(bin.size(), input.size())
        bin[:, :n] = input%2
        bin[:, n:] = torch.floor(input/2)
        return bin.to(self.dtype).to(self.device)

    def xyz(self, b_opt):
        '''inverse of rep() function, turn binary operator to normal representation'''
        if len(b_opt.size()) == 2:
            n = int(b_opt.size(1)/2)
            opt =  b_opt[:, :n] + 2*b_opt[:, n:]
        elif len(b_opt.size()) == 1:
            n = int(b_opt.size(0)/2)
            opt =  b_opt[:n] + 2*b_opt[n:]
        return opt.squeeze()

    def opt_prod(self, opt1, opt2):
        '''operator product'''
        opt1, opt2 = opt1.to(self.dtype).to(self.device), opt2.to(self.dtype).to(self.device)
        '''proudct rule of V4 group'''
        b_opt1, b_opt2 = self.rep(opt1), self.rep(opt2)
        b_opt = (b_opt1 + b_opt2)%2
        '''turn [[0, 0], [0, 1], [1, 0], [1, 1]] back to binary [0, 1, 2, 3]'''
        opt = self.xyz(b_opt)
        return opt.squeeze()
 
    def opts_prod(self, list_of_opts):
        '''product of list of operators'''
        s = list_of_opts.to(self.dtype).to(self.device)
        if len(s.size()) == 2:
            opt1 = s[0]
            a = 0
        else:
            a = 1
            opt1 = s[:, 0, :]
        b_opt1 = self.rep(opt1)
        for i in range(s.size(a)-1):
            if len(s.size()) == 2:
                opt2 = s[i+1]
            else:
                opt2 = s[:, i+1, :]
            b_opt2 = self.rep(opt2)
            b_opt1 = (b_opt1 + b_opt2)%2
        '''turn [[0, 0], [0, 1], [1, 0], [1, 1]] back to binary [0, 1, 2, 3]'''
        opt = self.xyz(b_opt1)
        return opt.squeeze()

    def confs_to_opt(self, confs, gs):
        '''
            get operators from the configurations and generators
            confs: (batch, m), configurations
            gs : (m, n), generators 
        '''
        confs, gs = confs.to(self.dtype).to(self.device), gs.to(self.dtype).to(self.device)
        if len(torch.tensor(confs.size())) <=1:
            confs = torch.unsqueeze(confs, dim=0)
        batch = confs.size(0)
        s = torch.tensor([gs.tolist()]*batch, device=self.device, dtype=self.dtype).permute((2, 0, 1))
        s = (s*confs).permute((1, 2, 0))
        opt = self.opts_prod(s)
        return opt 


    def commute(self, a, b, intype=('nor', 'nor')):
        '''
            calculate commutation relation of operators
        '''
        a, b = a.to(self.dtype).to(self.device), b.to(self.dtype).to(self.device)
        if len(a.size())<2:
            a = torch.unsqueeze(a, dim=0)
        if len(b.size())<2:
            b = torch.unsqueeze(b, dim=0)
        if intype == ('nor', 'nor'):
            I = torch.eye(a.size(1), device=self.device, dtype=self.dtype)
            bin_a = self.rep(a).squeeze()
            bin_b = self.rep(b).squeeze()
        elif intype == ('bin', 'bin'):
            I = torch.eye(int(a.size(1)/2), device=self.device, dtype=self.dtype)
            bin_a = a
            bin_b = b
        elif intype == ('nor', 'bin'):
            I = torch.eye(a.size(1))
            bin_a = self.rep(a).squeeze()
            bin_b = b
        elif intype == ('bin', 'nor'):
            I = torch.eye(int(a.size(1)/2), device=self.device, dtype=self.dtype)
            bin_a = a
            bin_b = self.rep(b).squeeze()
        
        Zero = torch.zeros_like(I, device=self.device, dtype=self.dtype)
        A = torch.cat([Zero, I], dim=0)
        B = torch.cat([I, Zero], dim=0)
        gamma = torch.cat([A, B], dim=1)
        return ((bin_a @ gamma @ bin_b.T)%2).squeeze()

    def row_echelon(self, M, b=None):
        
        M = deepcopy(M).to(self.dtype).to(self.device) 
        if b != None:
            b = deepcopy(b).to(self.dtype).to(self.device)
        '''Gauss elimination'''
        M1 = 0
        while (M-M1).sum() != 0:
            M1 = deepcopy(M)
            #print(M)
            for i in range(M.size(0)):
                if M[i].sum() == 0:
                    for j in range(i, M.size(0)):
                        if M[j].sum() !=0:
                            M[[i, j]] = M[[j, i]]
                    if b != None:
                        b[[i, j]] = b[[j, i]]
            for i in range(min(M.size(0), M.size(1))):
                if M[i,i] == 1:
                    None
                else:
                    for j in range(i, M.size(0)):
                                if M[j, i] == 1:
                                    M[[i, j]] = M[[j, i]]
                                    if b != None:
                                        b[[i, j]] = b[[j, i]]
            #print(M)
            c_idx = []
            for i in range(min(M.size(0), M.size(1))):
                if M[i,i] == 1:
                    c_idx.append(i)
                else:
                    if M[i].sum() !=0:
                        a = torch.nonzero(M[i])[0].item()
                        if a>i:
                            c_idx.append(a)
                    
           # print(c_idx)
            for i in range(min(M.size(0), M.size(1))):
                for j in range(len(c_idx)):
                    idx = c_idx[j]
                    #print(i, j, idx)
                    if M[i, idx] == 1 and j < i:
                        M[i, :] = abs(M[i, :] - M[j, :])
                        if b != None:
                            b[i] = abs(b[i] - b[j])
           
            for i in range(M.size(0)):
                if M[i].sum() == 0:
                    for j in range(i, M.size(0)):
                        if M[j].sum() !=0:
                            M[[i, j]] = M[[j, i]]
                    if b != None:
                        b[[i, j]] = b[[j, i]]
            
            for i in range(min(M.size(0), M.size(1))):
                if M[i,i] == 1:
                    None
                else:
                    for j in range(i, M.size(0)):
                        if M[j, i] == 1:
                            M[[i, j]] = M[[j, i]]
                            if b != None:
                                b[[i, j]] = b[[j, i]]
            for i in range(M.size(0)):
                for j in range(i+1, min(M.size(0), M.size(1))):
                    if M[i, j] == 1 and M[j, j] == 1:
                        M[i, :] = abs(M[i, :]- M[j, :])
                        if b != None:
                            b[i] = abs(b[i] - b[j])
            
        for i in range(M.size(0)):
            for j in range(M.size(0)):
                if j!= i and abs(M[i]-M[j]).sum()==0: 
                    M[j] = 0
                    if b != None:
                        b[j] = 0
        for i in range(M.size(0)):
            if M[i].sum() == 0:
                for j in range(i, M.size(0)):
                    if M[j].sum() !=0:
                        M[[i, j]] = M[[j, i]]
                if b != None:
                    b[[i, j]] = b[[j, i]]

        for i in range(min(M.size(0), M.size(1))):
            if M[i,i] == 1:
                None
            else:
                for j in range(i, M.size(0)):
                    if M[j, i] == 1:
                        M[[i, j]] = M[[j, i]]
                        if b != None:
                            b[[i, j]] = b[[j, i]]
        
        #print(M)
        if b != None:
            return M, b
        else:
            return M

    def rank(self, M):
        M_row_echelon = self.row_echelon(M)
        r = M_row_echelon.size(0)
        for i in range(M_row_echelon.size(0)):
            if M_row_echelon[i].sum() == 0:
                r -= 1
        return r
    def L_indep(self, M):
        A = M[0]
        r=1
        for i in range(1, M.size(0)):
            B = torch.vstack([A, M[i]])
            r1 = self.rank(B)
            #print(r)
            if r1 == r+1 :
                A = B
                r = r1
            else:
                None
            #print(A)
        return A
            
            
            

    def solve(self, M, b):
        M, b= self.row_echelon(M, b)
        n = M.size(1)
        m = M.size(0)
        if len(b.size())>1:
            x = torch.zeros(b.size(1), n, dtype=int)
            b = b.T
        else:
            x = torch.zeros(1, n, dtype=int)
            b = b.unsqueeze(dim=0)
        fv = n-m
        vars = []
        for i in range(n):
            if M[:, i].sum() == 0:
                vars.append(i)
        
        v=n
        c=0
        while c-v !=0 :
            c=v
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    x[:, idx] = b[:, i]
                    b[:, i] = abs(b[:, i]- x[:, idx])
                    for j in range(m):
                        if M[j, idx] == 1 and j!=i:
                            b[:, j] = abs(b[:, j]- x[:, idx])
                    M[:, idx] = 0           
        for i in range(n):
            for j in range(i, n):
                if j!=i and abs(M[:, j]- M[:, i]).sum() == 0 and M[:, j].sum() !=0 and j not in vars:
                    vars.append(j)

        M1 = deepcopy(M)
        M1[:, vars] = 0
        rvars = []
        for i in range(m-1, -1, -1):
            row = M1[i]
            k = row.sum()
            if k>1:
                k = k-1
                idx = row.nonzero().squeeze().tolist()
                for var in idx:
                    if var not in rvars and  var not in vars:
                        rvars.append(var)
                        idx.remove(var)
                        M1[:, var] = 0
                        break
                
                for var in idx:
                    if var not in vars and len(vars)<fv:
                        vars.append(var)
                        M1[:, var] = 0
            elif k==1:
                idx = M1[i].nonzero().squeeze().item()
                M1[:, idx] = 0
                rvars.append(idx)
        #     print(vars)
        #     print(rvars)
        # print(len(vars))
        # print(len(rvars))
        vars.sort()
        x[:, vars] = 0
        M[:, vars] = 0

        v=n-fv
        c=0
        while v !=0 and c<1000:
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    x[:, idx] = b[:, i]
                    b[:, i] = abs(b[:, i]- x[:, idx])
                    for j in range(m):
                        if M[j, idx] == 1 and j!=i:
                            b[:, j] = abs(b[:, j]- x[:, idx])
                    M[:, idx] = 0 
                    #print(i, idx, x)
            #print(v)
            c+=1
        return x.squeeze()

    def solution_space(self, M, b):
        M, b = self.row_echelon(M, b)
        #print(M)
        n = M.size(1)
        m = M.size(0)
        x = torch.zeros(n, dtype=int)
        
        fv = n-m
        vars = []
        for i in range(n):
            if M[:, i].sum() == 0:
                vars.append(i)
        
        v=n
        c=0
        while c-v !=0 and v>0:
            c=v
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    x[idx] = b[i]
                    b = abs(b-(M[:, idx]*x[idx])%2)
                    M[:, idx] = 0           

        for i in range(n):
            for j in range(i, n):
                if j!=i and abs(M[:, j]- M[:, i]).sum() == 0 and M[:, j].sum() !=0 and j not in vars:
                    vars.append(j)

        M1 = deepcopy(M)
        while len(vars) < fv:
            c_max = torch.argmax(torch.sum(M1, dim=0)).item()
            if c_max not in vars:
                vars.append(c_max)
            M1[:, c_max] = 0
            for i in range(m):
                if M1[i].sum() == 1:
                    idx = M1[i].nonzero().squeeze()
                    M1[:, idx] = 0

        vars.sort()

        space = torch.vstack([x]*2**len(vars))
        b = torch.vstack([b]*2**len(vars))
        config = exact_config(D=len(vars), N=2**len(vars)).long()
        #print(config)
        for i in range(2**len(vars)):
            space[i][vars] = config[i]
        for j in range(m):   
            if M[j].sum() != 0:
                b[:, j] = (b[:, j] - (M[j, vars]@config.T)%2)%2
                M[j, vars] = 0
                
        v=n
        c=0
        while c-v !=0 :
            c=v
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    space[:, idx] = b[:, i]
                    b[:, i] = abs(b[:, i]-space[:, idx])
                    for j in range(m):
                        if M[j, idx] == 1 and j!=i:
                            b[:, j] = abs(b[:, j]-space[:, idx])
                    M[:, idx] = 0 
                 
        return space
    
    def kernel(self, M):
        M = self.row_echelon(M)
        #print(M)
        n = M.size(1)
        m = M.size(0)
        b = torch.zeros(m).long()
        if len(b.size())>1:
            x = torch.zeros(n, b.size(1), dtype=int)
        else:
            x = torch.zeros(n, dtype=int)
        
        fv = n-m
        vars = []
        for i in range(n):
            if M[:, i].sum() == 0:
                vars.append(i)
        
        v=n
        c=0
        while c-v !=0 and v>0:
            c=v
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    x[idx] = b[i]
                    b = abs(b-(M[:, idx]*x[idx])%2)
                    M[:, idx] = 0           

        for i in range(n):
            for j in range(i, n):
                if j!=i and abs(M[:, j]- M[:, i]).sum() == 0 and M[:, j].sum() !=0 and j not in vars:
                    vars.append(j)

        # M1 = deepcopy(M)
        # while len(vars) < fv:
        #     c_max = torch.argmax(torch.sum(M1, dim=0)).item()
        #     if c_max not in vars:
        #         vars.append(c_max)
        #     M1[:, c_max] = 0
        #     for i in range(m):
        #         if M1[i].sum() == 1:
        #             idx = M1[i].nonzero().squeeze()
        #             M1[:, idx] = 0
        
        M1 = deepcopy(M)
        M1[:, vars] = 0
        rvars = []
        for i in range(m-1, -1, -1):
            row = M1[i]
            k = row.sum()
            if k>1:
                k = k-1
                idx = row.nonzero().squeeze().tolist()
                for var in idx:
                    if var not in rvars and  var not in vars:
                        rvars.append(var)
                        idx.remove(var)
                        M1[:, var] = 0
                        break
                
                for var in idx:
                    if var not in vars and len(vars)<fv:
                        vars.append(var)
                        M1[:, var] = 0
            elif k==1:
                idx = M1[i].nonzero().squeeze().item()
                M1[:, idx] = 0
                rvars.append(idx)

        vars.sort()

        #print(vars)
        space = torch.vstack([x]*len(vars))
        b = torch.vstack([b]*len(vars))
        config = torch.eye(len(vars)).long()
        #print(config)
        for i in range(len(vars)):
            space[i][vars] = config[i]
        for j in range(m):   
            if M[j].sum() != 0:
                b[:, j] = (b[:, j] - (M[j, vars]@config.T)%2)%2
                M[j, vars] = 0
                
        v=n
        c=0
        while c-v !=0 :
            c=v
            for i in range(m):
                if M[i].sum() == 1:
                    v=v-1
                    idx = M[i].nonzero().squeeze()
                    space[:, idx] = b[:, i]
                    b[:, i] = abs(b[:, i]-space[:, idx])
                    for j in range(m):
                        if M[j, idx] == 1 and j!=i:
                            b[:, j] = abs(b[:, j]-space[:, idx])
                    M[:, idx] = 0 
        if b.sum() == 0:
            r = []
            for i in range(space.size(0)):
                if space[i, vars].sum() == 1:
                    r.append(i)
            space = space[r]
        return space

    def Schmidt(self, N):
        n = N.size(0)
        N1 = deepcopy(N)
        anti_pairs = []
        i = 0
        while self.commute(N1, N1).sum() !=0:
            com = self.commute(N1[i], N1)
            if com.sum() !=0 :
                #print(i)
                #print(com)
                idx = com.nonzero()[0].item()
                N1[[0, i]] = N1[[i, 0]]
                N1[[1, idx]] = N1[[idx, 1]]
                N2 = N1[[0, 1]]
                anti_pairs.append(N2)
                N1 = N1[2:]
                
                com1 = self.commute(N2[0], N1)
                #print(com1)
                idx1 = com1.nonzero().squeeze()
                #print(idx1)
                N1[idx1] = self.opt_prod(N1[idx1], N2[1])
                #print(self.commute(N2[0], N1))

                com2 = self.commute(N2[1], N1)
                #print(com2)
                idx2 = com2.nonzero().squeeze()
                N1[idx2] = self.opt_prod(N1[idx2], N2[0])
                #print(self.commute(N2[1], N1))
                i=0
            else:
                i+=1
        return anti_pairs

                
                    

if __name__ == '__main__':
    m2 = mod2()
    print('random initial a binary matrix (m, n) with rank m and m<n')
    def Matrix(m, n):
        M = torch.randint(0, 2, (m, n))
        while m2.rank(M) != m:
            M = torch.randint(0, 2, (m, n))
        return M
    m, n = 5, 6
    M = Matrix(m, n)
    print(M)
    print('row echelon M :')
    print(m2.row_echelon(M))

    print('free var number :', n-m)

    b1 = torch.eye(m).long()
    print('solution :')
    x0 = m2.solve(M, b1)
    print(x0)
    print((M@x0.T)%2)

    print('initial a result vector b:')
    b = torch.ones(m).long()
    print(b)
    
    x = m2.solution_space(M, b)
    print('solution space :')
    print(x)
    print('test!!!')
    print((M@x.T)%2)
    print('space rank :', m2.rank(x))


    
    
    # A = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 1],
    #     [0, 1, 0, 0, 0, 1, 1, 0],
    #     [1, 0, 1, 0, 0, 1, 1, 1],
    #     [1, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 0, 1, 0, 0, 1],
    #     [0, 1, 0, 0, 1, 1, 1, 0],
    #     [1, 0, 1, 0, 1, 1, 1, 1],
    #     [1, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 1],
    #     [0, 1, 0, 1, 0, 1, 1, 0],
    #     [1, 0, 1, 1, 0, 1, 1, 1],
    #     [1, 0, 1, 1, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 1, 0, 0, 1],
    #     [0, 1, 0, 1, 1, 1, 1, 0],
    #     [1, 0, 1, 1, 1, 1, 1, 1]])
    # print(m2.row_echelon(A))
    
    
    
    