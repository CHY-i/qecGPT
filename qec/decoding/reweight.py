from args import args
import torch
import torch.nn.functional as F
import time
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Errormodel, mod2, Loading_code, read_code, btype
torch.backends.cudnn.enable =True

def reweight(n_s, k, van, syndrome, Errormodel, gs, device, dtype):
     
    marginals = torch.zeros(2**(2*k),device=device, dtype=dtype)
    for i in range(2**(2*k)):
        condition = syndrome*2-1
        l = (btype(i, 2*k)*2-1).to(device).to(dtype)
        condition = torch.hstack([condition, l]).squeeze()
        #print(condition.size())
        s = van.partial_samples(n_s=n_s, condition=condition, device=device, dtype=dtype)
        #print(s)
        x = (s + 1)/2
        #print(x)
        logq = van.partial_logp(s, m=condition.size(0))
        opts = mod2.confs_to_opt(x, gs)
        logp = Errormodel.log_probability(opts, device, dtype)
        mi = torch.mean(torch.exp(logp-logq))
        marginals[i] = mi
    #print(marginals)
    return btype(torch.argmax(marginals), 2*k)


def reweight1(n_s, k, van, syndrome, Errormodel, gs, device, dtype): 
    condition = syndrome*2-1
    s = van.partial_samples(n_s=n_s, condition=condition, device=device, dtype=dtype)
    with torch.no_grad():
        logq = van.partial_logp(s, m=condition.size(0))
        x = (s + 1)/2
        opts = mod2.confs_to_opt(x, gs)
        logp = Errormodel.log_probability(opts, device, dtype)
        Z = torch.exp(torch.mean(logp - logq))
    
    return 
  
# def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
#     if n_type =='made':
#         condition = syndrome*2-1
#         x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
#     elif n_type == 'trade':
#         condition = syndrome
#         x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
#     x = x[:, m:m+2*k]
#     return x
save = args.save
'''Hyper Paras:'''
device, dtype = args.device, torch.float64,
trials = args.trials
'''Code Paras:'''
c_type, d, k, seed = args.c_type, args.d, args.k, args.seed
'''Net Paras:'''
l_type, n_type = 'fkl', 'made'
er = args.er
'''seed for sampling errors:'''
error_seed = args.error_seed


mod2 = mod2(device=device, dtype=dtype)

'''Loading Code'''
info = read_code(d, k, seed, c_type=c_type)
Code = Loading_code(info)

e1 = Code.pure_es 
    
for i in range(Code.m):
    conf = mod2.commute(e1[i], e1)
    idx = conf.nonzero().squeeze()
    #print(idx)
    sta = Code.g_stabilizer[idx]
    e1[i] = mod2.opts_prod(torch.vstack([e1[i], sta]))

g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer])

'''Loading Net'''
path = abspath(dirname(__file__))+'/net/'+l_type+'_'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_mid.pt'.format(d, k, seed, er)
net = torch.load(path)

if n_type == 'made':
    van = MADE(2*Code.n, 0, 1).to(device)
    van.deep_net = net.to(device)
elif n_type =='trade':
    van = net.to(device)
    van.device = device

if k == 1 :
    error_rate = torch.linspace(0.01, 0.368, 19)
else:
    error_rate = torch.linspace(0.01, 0.25, 20)
lo_rate = []
#error_rate = [0.189]
tt = torch.zeros(len(error_rate))
for i in range(len(error_rate)):
    '''generate error'''
    E = Errormodel(error_rate[i], e_model='depolarized')#error_rate[i]
    errors = E.generate_error(Code.n,  m=trials, seed=error_seed)
    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)
    #print(syndrome)
    correct_number = 0
    t = 0
    for j in range(trials):
        '''sampling'''
        t1 = time.time()
        lconf = reweight(n_s=10000, k=k, van=van, syndrome=syndrome[j], Errormodel=E, gs=g, device=device, dtype=dtype)
        #lconf1 = reweight1(n_s=100, k=k, van=van, syndrome=syndrome[j], Errormodel=E, gs=g, device=device, dtype=dtype)
        # print(lconf)
        # print(lconf1)

        '''correction'''
        l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
        recover = mod2.opt_prod(pe[j], l)
        check = mod2.opt_prod(recover, errors[j])
        commute = mod2.commute(check, Code.logical_opt).sum()
        if commute == 0:
            correct_number+=1
        logical_error_rate = 1-correct_number/trials
        t2 = time.time()
        t = t+(t2-t1)
        #print(correct_number)
    ta = t/trials
    print(ta)
    tt[i] = ta
    print(logical_error_rate)

    
    lo_rate.append(logical_error_rate)
if save == True:
    torch.save((lo_rate), abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_'.format(d, k, seed) +n_type+'_reweight_{}_{}1.pt'.format(error_seed, er))
print(lo_rate)
print(tt.mean().item(), tt.std().item())