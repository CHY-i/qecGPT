from args import args
import torch
import time
import sys
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Errormodel, mod2, Loading_code, read_code
torch.backends.cudnn.enable =True

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type =='made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type == 'trade':
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]
    return x

'''Hyper Paras:'''
if args.dtype == 'float32':
    dtype = torch.float32
elif args.dtype == 'float64':
    dtype = torch.float64
device = args.device
trials = args.trials
'''Code Paras:'''
c_type, d, k, seed = args.c_type, args.d, args.k, args.seed
'''Net Paras:'''
l_type, n_type, e_model = 'fkl', args.n_type, args.e_model
save = args.save



if e_model == 'depolarized':
    er = args.er
    '''seed for sampling errors:'''
    error_seed = args.error_seed
    path = abspath(dirname(__file__))+'/net/'+l_type+'_'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_mid.pt'.format(d, k, seed, er)
    
    if k == 1 and c_type=='sur':
        error_rate = torch.linspace(0.01, 0.368, 19)
    elif k == 1 and c_type=='rsur':
        error_rate = torch.linspace(0.01, 0.368, 19)
    elif k == 1 and c_type=='3d':
        error_rate = torch.linspace(0.01, 0.368, 19)
    elif k==2 and c_type=='tor':
        error_rate = torch.linspace(0.01, 0.368, 19)
    else:
        error_rate = torch.linspace(0.01, 0.25, 20)
        
elif e_model == 'ising':
    beta = args.beta
    h = args.h
    path = abspath(dirname(__file__))+'/net/'+l_type+'_'+n_type+'_'+c_type+'_d{}_k{}_seed{}_ising{}_h{}_mid.pt'.format(d, k, seed, beta, h)
    beta_list = [2, 1, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]


# error_rate = torch.tensor([0.03,0.03366055, 0.03776776, 0.04237613, 0.0475468, 0.05334838,
#  0.05985787, 0.06716163,0.07535659, 0.08455149, 0.09486833, 0.10644402,
#  0.11943215, 0.13400508,0.15035617, 0.1687024,  0.1892872,  0.21238373,
#  0.23829847, 0.26737528, 0.3])
'''Loading Code'''
info = read_code(d, k, seed, c_type=c_type)
Code = Loading_code(info)

'''Loading Net'''
net = torch.load(path)
if n_type == 'made':
    van = MADE(2*Code.n, 0, 1).to(device).to(dtype)
    van.deep_net = net.to(device).to(dtype)
elif n_type =='trade':
    van = net.to(device).to(dtype)
    van.device = device
    van.dtype = dtype

mod2 = mod2(device=device, dtype=dtype)





lo_rate = []
for i in range(len(error_rate) if e_model == 'depolarized' else len(beta_list)):
    '''generate error'''
    if e_model == 'depolarized':
        E = Errormodel(error_rate[i], e_model='depolarized')#error_rate[i]
        errors = E.generate_error(Code.n,  m=trials, seed=error_seed)
    elif e_model == 'ising':
        E = Errormodel()
        errors, _ = E.ising_error(n_s=trials, n=Code.n, beta=beta_list[i], h=h)

    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)

    '''forward to get configs'''
    
    lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)
    #print(lconf)

    '''correction'''
    l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
    recover = mod2.opt_prod(pe, l)
    check = mod2.opt_prod(recover, errors)
    commute = mod2.commute(check, Code.logical_opt)
    fail = torch.count_nonzero(commute.sum(1))
    logical_error_rate = fail/trials
    print(logical_error_rate)
#         if commute == 0:
#             correct_number+=1
#         print(correct_number, j+1)    
#     logical_error_rate = 1-correct_number/trials
#     print(logical_error_rate)
    lo_rate.append(logical_error_rate.cpu().item())
print(lo_rate)
if save == True:
    if e_model == 'depolarized':
        path = abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_'.format(d, k, seed) +n_type+'_forward_{}_{}_mid.pt'.format(error_seed, er)
        if exists(path):
            print('exists')
        else:
            torch.save((lo_rate), path)
    elif e_model == 'ising':
        path = abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_'.format(d, k, seed) +n_type+'_forward_ising_{}_{}_mid.pt'.format(beta, h)
        if exists(path):
            print('exists')
        else:
            torch.save((lo_rate), path)
            

