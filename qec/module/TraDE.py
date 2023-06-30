import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, n).reshape(n, 1)
        pos_embedding = torch.zeros((n, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)
        positions = torch.arange(n)
        self.register_buffer('positions', positions)

    def forward(self, x):
        return x + self.positional_embedding(self.positions)


class TraDE(nn.Module):
    """
    Transformers for density estimation or stat-mech problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.dropout = kwargs['dropout']

        self.fc_in = nn.Embedding(2, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        # self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask, diagonal=0)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))#.masked_fill(self.mask == 1, float(0.0))
        #self.mask[0, 0] = 0
    def forward(self, x):
        x = torch.cat((torch.ones(x.size(0), 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.to(int)))  # (batch_size, n, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)
        return torch.sigmoid(self.fc_out(x)).squeeze(2)

    def log_prob(self, x):
        x_hat = self.forward(x)
        log_prob = torch.log(x_hat) * x + torch.log(1 - x_hat) * (1 - x)
        return log_prob.sum(dim=1)
    
    def samples(self, batch_size):
        samples = torch.zeros(batch_size, self.n, device=self.device, dtype=torch.float64)#torch.randint(0, 2, size=(batch_size, self.n), dtype=torch.float64, device=self.device)
        for i in range(self.n):
            x_hat = self.forward(samples)
            samples[:, i] = torch.bernoulli(x_hat[:, i])
        return samples

    def partial_samples(self, n_s, condition, device, dtype):
        with torch.no_grad():
            m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = torch.vstack([condition]*n_s)
            for i in range(self.n-m):
                s_hat = self.forward(x)
                x[:, m+i] = torch.bernoulli(s_hat[:, m+i])
        return x
    
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
                x[:, m+i] = torch.floor(2*s_hat[:, m+i])
        return x

    def test(self):
        res = []
        s0 = torch.ones(1, kwargs_dict['n'], requires_grad=True).to(kwargs_dict['device']).int()
        s = F.relu(self.fc_in(s0))
        s = self.positional_encoding(s)
        s.retain_grad()
        for k in range(self.n):
            x = self.encoder(s, mask=self.mask)
            x = torch.sigmoid(self.fc_out(x)).squeeze(2)
            loss = x[0, k]
            loss.backward(retain_graph=True)
            grad = s.grad.sum(2)
            print(s.grad)
            depends = (grad[0].numpy() != 0).astype(np.uint8)
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
    L, dim=10, 2
    beta = 0.8
    device='cuda:0'
    dtype=torch.float64
    n = L if dim==1 else L**2

    batch, epoch, lr = 1000, 5000, 0.001

    kwargs_dict = {
        'n': L if dim==1 else L**2,
        'd_model': n,
        'd_ff': int(n/2),
        'n_layers': 2,
        'n_heads': 2,
        'device': device,
        'dropout':0,#'cuda:5'
    }


    Is = Ising(L, beta=beta, dim=dim, device=device, dtype=dtype)
    if dim==2:
        Ex = kacward(L, Is.J, beta)
        Fe = -Ex.lnZ/(beta*L*L)
    else:
        Ex = exact(Is.graph, Is.J, beta, device=device,dtype=dtype,seed=0)
        Fe = -Ex.lnZ_fvs()/(beta*L)
    print(Fe)

    T = TraDE(**kwargs_dict).to(kwargs_dict['device'])
 
    optimizer = torch.optim.Adam(T.parameters(), lr=lr)#torch.optim.SGD(van.parameters(), lr=LR, momentum=0.9)#
    
    for i in range(epoch):
        with torch.no_grad():
            sample = T.samples(batch)
            s = sample.reshape(sample.size(0), -1)*2 -1
            E = Is.energy(s)
        logp = T.log_prob(sample)
        with torch.no_grad():
            loss = beta * E + logp
        loss_reinforce = torch.mean((loss - loss.mean()) * logp)
        optimizer.zero_grad()
        loss_reinforce.backward()
        optimizer.step()
        Fq = (loss.mean()/(beta*n)).cpu().item()
    
        #Fq_his.append(Fq)

        print(i, Fq, abs((Fq-Fe)/Fe), loss.std())
    
    

    
    
    