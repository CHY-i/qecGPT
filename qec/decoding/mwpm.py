from args import args
from pymatching import Matching
import numpy as np
import torch
import time
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, Hx_Hz, Errormodel, mod2

d, k, seed, c_type = args.d, args.k, args.seed, args.c_type
trials = args.trials
device, dtype = 'cpu', torch.float64
e_model = args.e_model
error_seed = args.error_seed
mod2 = mod2(device=device, dtype=dtype)

info = read_code(d, k, seed, c_type=c_type)
code = Loading_code(info)
n = code.n
PCM = code.PCM.cpu().numpy()
#print(PCM)
hx, hz = Hx_Hz(code.g_stabilizer)
hx, hz = hx.cpu().numpy(), hx.cpu().numpy()
#print(hx)
#print(hz)

l1 = mod2.rep(code.logical_opt).int().numpy()
l = np.zeros_like(l1)
l[:, :n], l[:, n:] = l1[:, n:], l1[:, :n]
#print(l)


if e_model == 'depolarized':
    if k == 1 :
        error_rate = torch.linspace(0.01, 0.368, 19)
    elif k==2 and c_type=='tor':
        error_rate = torch.linspace(0.01, 0.368, 19)
    else:
        error_rate = torch.linspace(0.01, 0.25, 20)
elif e_model == 'ising':
    beta = args.beta
    h = args.h
    beta_list = [2, 1, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    
L = []
#error_rate = torch.linspace(0, 0.3, 30)
tt = torch.zeros(len(error_rate) if e_model == 'depolarized' else len(beta_list))
for i in range(len(error_rate) if e_model == 'depolarized' else len(beta_list)):
    '''generate error'''
    if e_model == 'depolarized':
        E = Errormodel(error_rate[i], e_model='depolarized')#error_rate[i]
        errors = E.generate_error(code.n,  m=trials, seed=error_seed)
        weights = torch.ones(2*code.n)*torch.log((1-error_rate[i])/error_rate[i])
        Decoder = Matching(PCM, weights=weights)
    elif e_model == 'ising':
        E = Errormodel()
        errors, w = E.ising_error(n_s=trials, n=code.n, beta=beta_list[i], h=h)
        weights = torch.hstack([(args.trials-w)/2]*2)/args.trials
        #print(weights)
        Decoder = Matching(PCM, weights=weights)

    syndrome = mod2.commute(errors, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)
    error = mod2.rep(errors).squeeze().int().numpy()
    syndrome = syndrome.numpy()

    correct_number = 0
    t = 0
    for j in range(trials):
        e = error[j]
        s = syndrome[j]

        t1 = time.time()
        #print(s)
        recover = Decoder.decode(s)
        check = (e + recover)%2
        s = np.sum((check @ l.T) %2)
        t2 = time.time()
        t = t+(t2-t1)
        if s == 0:
            correct_number+=1
        
    lorate = 1 - correct_number/trials
    ta = t/trials
    print(lorate)
    print(ta)
    tt[i] = ta
    L.append(lorate)
print(L)
print(tt.mean().item(), tt.std().item())    




