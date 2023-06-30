from copy import deepcopy
import torch
import opt_einsum as oe
import math
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, mod2, CodeTN, Errormodel, exact_config, Abstractcode


def Multi_MLD_conditional(trials, code, error_rate, device, dtype, seed):
    k = code.n-code.m
    
    E = Errormodel(e_rate=error_rate)
    error = E.generate_error(n=code.n, m=trials, seed=seed)
    syndrome = mod2.commute(error, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)

    L = code.logical_opt
    conf1 = exact_config(1, 2)
    #print(conf1)
    lrecover = torch.zeros(trials, code.n, device=device,dtype=dtype)

    e1 = deepcopy(pe)
    for i in range(2*k):
        margin_cos = []
        l = mod2.confs_to_opt(confs=conf1,gs=L[i].unsqueeze(0))
        #print(l)
        lprime = L[i+1:]
        #print(lprime)
        codeprime = Abstractcode(torch.vstack([code.g_stabilizer, lprime]), complete=False)
        TN = CodeTN(codeprime, multi=True, device=device, dtype=dtype)
        eq = TN.ein_eq()
        for j in range(2):
            e = mod2.opt_prod(e1, l[[j]*pe.size(0)])
            tensors, lm = TN.generate_tensors(e, E.single_p)
            margin_cos.append(oe.contract(eq, *tensors, optimize='auto').unsqueeze(dim=1))
        cos = torch.cat(margin_cos, dim=1)
        index = torch.argmax(cos, dim=1)
        e1 = mod2.opt_prod(e1, l[index])
        lrecover = mod2.opt_prod(lrecover, l[index])
    erecover = mod2.opt_prod(pe, lrecover)
    echeck =  mod2.opt_prod(erecover, error)
    check = mod2.commute(echeck, L).sum(dim=1)
    #print(check)
    fail_number = torch.count_nonzero(check)
    print(fail_number.item())


    logical_error_rate = fail_number/trials
    print(logical_error_rate.item())
    return logical_error_rate.item()
if __name__ =='__main__':
    import time
    device, dtype = 'cuda:1', torch.float64
    trials = 10000
    c_type='sur'
    d, k, seed = 5, 5, 0
    info = read_code(d, k, seed, c_type=c_type)
    code = Loading_code(info)
    mod2 = mod2(device=device, dtype=dtype)

    error_seed = 10000
    error_rate = torch.linspace(0, 0.3, 30)#[0.15]#
    lo_rate = []
    t=0
    for i in range(len(error_rate)):
        t0 = time.time()
        lrate = Multi_MLD_conditional(trials=trials, code=code, error_rate=error_rate[i], device=device, dtype=dtype, seed=error_seed)
        t1 = time.time()
        lo_rate.append(lrate)
        t += t1-t0
        print(t1-t0)
    t = t/len(error_rate)
    print(lo_rate)
    torch.save((lo_rate, t), abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_conditional_{}.pt'.format(d, k, seed, error_seed))