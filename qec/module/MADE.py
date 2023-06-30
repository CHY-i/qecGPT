import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, n, in_features, out_features, self_connection=True, bias=True, residual=False):
        '''n: 自旋个数,
           n*in: 总的输入个数,
           n*out: 总的输出个数,
         '''
        super(MaskedLinear, self).__init__(n * in_features, n * out_features, bias)
        #定义一个名为mask个的buffer     
        if self_connection:
            self.register_buffer('mask', torch.tril(torch.ones(n, n)))#注意 pytorch中是用行向量乘W.T定义的线性运算
        else:
            self.register_buffer('mask', torch.tril(torch.ones(n, n), diagonal=-1))
        self.mask = torch.cat([self.mask] * in_features, dim=1)
        self.mask = torch.cat([self.mask] * out_features, dim=0)
        self.weight.data *= self.mask
        if n !=1 :
            self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
        self.residual = residual
    def forward(self, input):
        if self.residual == False:
            return F.linear(input, self.weight*self.mask, self.bias)
        elif self.residual == True:
            return F.linear(input, self.weight*self.mask, self.bias) + input

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.positional_encoding = nn.Linear(n, n)
        positions = torch.arange(n)
        self.register_buffer('positions', positions)

    def forward(self, x):
        return x + self.positional_encoding(self.positions)


class MADE(nn.Module):
    def __init__(self, n, depth, width, activator='tanh'):
        '''
            n: 自旋个数，为网络的输入输出神经元个数
            depth: 网络深度
        '''
        super(MADE, self).__init__()
        self.n = n
        self.depth = depth
        self.n_hiddens = depth-1
        self.width = width
        if activator=='tanh':
            self.activator = nn.Tanh()
        elif activator=='relu':
            self.activator = nn.ReLU()
        elif activator=='sigmoid':
            self.activator = nn.Sigmoid()

        self.construction(width, depth)


    def construction(self, width, depth):
        n = self.n
        self.deep_net = []
        self.deep_net.extend([
            MaskedLinear(n, 1, 1 if depth==0 and width==1 else width, False), 
            self.activator,
            ])
        for i in range(depth):
            self.deep_net.extend([
                MaskedLinear(n, width, width, True, True, residual=False),
                self.activator,
                ])
        if width != 1:
            self.deep_net.extend([
                MaskedLinear(n, width, 1, True, True),
                self.activator,
                ])
        self.deep_net.pop()
        self.deep_net.extend([nn.Sigmoid(),])
        self.deep_net = nn.Sequential(*self.deep_net)
        
            

    def forward(self, x):
        #if self.residual==False:
        return self.deep_net(x)
        # elif self.residual==True:
        #     x1 = self.deep_net(x)
        #     RLinear = MaskedLinear(self.n, self.width, 1, True, True).to(x.device, x.dtype)
        #     x2 = F.sigmoid(RLinear.forward(x1, x))
        #     return x2

    def partial_forward(self, n_s, condition, device, dtype, k=1):
        with torch.no_grad():
            if n_s >1 :
                m = condition.size(1)
            else:
                m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = condition
            for i in range(2*k):
                s_hat = self.forward(x)
                x[:, m+i] = torch.floor(2*s_hat[:, m+i]) * 2 - 1
        return x
    
    def prob(self, condition, device, dtype):
        with torch.no_grad():
            m = condition.size(0)
            x = torch.zeros(self.n, device=device, dtype=dtype)
            x[:m] = condition
            s_hat = self.forward(x)
        return s_hat[m+1]


    def samples(self, n_s, n, device='cpu', dtype=torch.float64, max_sampling=False):
        s = torch.zeros(n_s, n, device=device, dtype=dtype)
        for i in range(n):
            s_hat = self.forward(s)
            if max_sampling == True:
                s[:, i] = torch.floor(s_hat[:, i]*2) *2 -1
            else:
                s[:, i] = torch.bernoulli(s_hat[:, i]) * 2 - 1
        return s

    def partial_samples(self, n_s, condition, device, dtype):
        with torch.no_grad():
            m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = torch.vstack([condition]*n_s)
            for i in range(self.n-m):
                s_hat = self.forward(x)
                # print(i, s_hat.shape, s_hat.device, s.shape, s.device)
                #if i >= m:
                x[:, m+i] = torch.bernoulli(s_hat[:, m+i]) * 2 - 1
        return x
    

    def log_prob(self, samples):
        s = self.forward(samples)
        mask = (samples + 1)/2
        mask = mask.view(mask.shape[0], - 1)
        log_p = (torch.log(s) * mask + torch.log(1 - s) * (1 - mask)).sum(dim=1)
        return log_p
    
    def partial_logp(self, samples, m):
        s = self.forward(samples)
        samples_prime = samples[:, m:]
        s_prime = s[:, m:]
        mask = (samples_prime + 1)/2
        mask = mask.view(mask.shape[0], - 1)
        log_p = (torch.log(s_prime) * mask + torch.log(1 - s_prime) * (1 - mask)).sum(dim=1)
        return log_p

    def energy_s(self, J, samples):
        T = samples@J@samples.T/2 #能量是该矩阵主对角线的值
        E = torch.diag(T)
        return E


    def test(self):
        res = []
        rng = np.random.RandomState(14)
        x = (rng.rand(1, self.n) > 0.5).astype(np.float64)
        for k in range(self.n):
            xtr = Variable(torch.from_numpy(x), requires_grad=True)
            xtrhat = self.forward(xtr)
            loss = xtrhat[0, k]
            loss.backward()
            
            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % self.n not in depends_ix
            
            res.append((len(depends_ix), k, depends_ix, isok))
        
            # pretty print the dependencies
            res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %70s : %s" % (k, ix, "OK" if isok else "NOTOK"))


    




if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname
    sys.path.append(abspath(dirname(__file__)).strip('module'))
    print(abspath(dirname(__file__)).strip('module'))
    from module.exact import exact, kacward
    from module.physical_model import Ising

    L = 4
    beta = 0.4406868
    device='cuda:0'
    dtype=torch.float64

    Is = Ising(L, beta=beta, dim=2, device=device, dtype=dtype)
    Ex = kacward(L, Is.J, beta)
    Fe = -Ex.lnZ/(beta*L*L)
    print(Fe)
    batch, epoch, lr = 10000, 2000, 0.01
    depth, width = 0, 1
    van = MADE(L*L, depth, width).to(device)

    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#torch.optim.SGD(van.parameters(), lr=LR, momentum=0.9)#
    
    for i in range(epoch):
        with torch.no_grad():
            sample = van.samples(batch, L*L, device=device, dtype=dtype, max_sampling=False)
            s = sample.reshape(sample.size(0), -1)#*2 -1
            E = Is.energy(s)
        logp = van.log_prob(sample)
        with torch.no_grad():
            loss = beta * E + logp
        loss_reinforce = torch.mean((loss - loss.mean()) * logp)
        optimizer.zero_grad()
        loss_reinforce.backward()
        optimizer.step()
        Fq = (loss.mean()/(beta*L*L)).cpu().item()
        #Fq_his.append(Fq)

        print(i, Fe, Fq, abs((Fq-Fe)/Fe))