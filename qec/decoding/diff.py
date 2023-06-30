from time import time
import torch
import opt_einsum as oe
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, mod2, CodeTN, Errormodel, exact_config



def coset_probability(code, errors, error_rate, device, dtype):

    TN = CodeTN(code, multi=True, device=device, dtype=dtype)
    eq = TN.ein_eq()

    E = Errormodel(e_rate=error_rate)
    syndrome = mod2.commute(errors, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)
    
    L = code.logical_opt
    conf = exact_config(len(L), 2**len(L), device=device, dtype=dtype)
    lopt = mod2.confs_to_opt(confs=conf, gs=L)

    cos = []
    elist = []
    for i in range(2**len(L)):
        l = lopt[[i]*pe.size(0)]
        e = mod2.opt_prod(pe, l)
        elist.append(e.unsqueeze(dim=0))
        tensors, lm = TN.generate_tensors(e, E.single_p)
        cos.append(oe.contract(eq, *tensors, optimize='auto').unsqueeze(dim=1))

    cos = torch.cat(cos, dim=1)
    return cos

if __name__ =='__main__':
    import time
    device, dtype = 'cuda:1', torch.float64
    trials = 10000
    c_type='sur'
    d, k, seed = 11, 1, 0
    error_seed = 10000
    
    info = read_code(d, k, seed, c_type=c_type)
    code = Loading_code(info)
    mod2 = mod2(device=device, dtype=dtype)


    er = 0.189
    error_rate = torch.linspace(0.01, 0.25, 20)
    print(error_rate)
    
    t=0
    if d>7:
        diff = torch.zeros(len(error_rate), device=device, dtype=dtype)
    else:
        diff = []
    for i in range(len(error_rate)):
        E = Errormodel(error_rate[i])
        errors = E.generate_error(n=code.n, m=trials, seed=error_seed)
        if d>7:
            
            for j in range(10):
                es = errors[j*1000:(j+1)*1000]
                cp0 = coset_probability(code=code, errors = es, error_rate=er, device=device, dtype=dtype)

                cp = coset_probability(code=code, errors = es, error_rate=error_rate[i], device=device, dtype=dtype)
                
                a = torch.argmax(cp0, dim=1)
                b = torch.argmax(cp, dim=1)
                diff[i] += torch.nonzero(a-b).size(0)/trials
            print(diff)
        else:
            cp0 = coset_probability(code=code, errors = errors, error_rate=er, device=device, dtype=dtype)

            cp = coset_probability(code=code, errors = errors, error_rate=error_rate[i], device=device, dtype=dtype)
            
            a = torch.argmax(cp0, dim=1)
            b = torch.argmax(cp, dim=1)
            diff.append(torch.nonzero(a-b).size(0)/trials)
print(diff)