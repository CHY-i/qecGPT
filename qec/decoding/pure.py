from args import args
import torch
import time
import sys
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Errormodel, mod2, Loading_code, read_code

device, dtype = 'cuda:5', torch.float64
trials = 10000
c_type='sur'
d, k, seed = 3, 1, 0
t_p = 10
error_seed = 10000

info = read_code(d, k, seed, c_type=c_type)
code = Loading_code(info)
mod2 = mod2(device=device, dtype=dtype)

error_rate = torch.linspace(0.01, 0.368, 19)
lo_rate = []

for i in range(len(error_rate)):
    E = Errormodel(e_rate=error_rate[i])
    error = E.generate_error(n=code.n, m=trials, seed=seed)
    syndrome = mod2.commute(error, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)

    recover = pe
    check = mod2.opt_prod(recover, error)
    commute = mod2.commute(check, code.logical_opt)
    fail = torch.count_nonzero(commute.sum(1)).cpu().item()
    logical_error_rate = fail/trials
    lo_rate.append(logical_error_rate)

print(lo_rate)