from time import time
import torch
import opt_einsum as oe
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, mod2, CodeTN, Errormodel, exact_config

def MLD(d, trials, code, error_rate, device, dtype, seed):
    TN = CodeTN(code, multi=False, device=device, dtype=dtype)
    eq = TN.ein_eq()

    E = Errormodel(e_rate=error_rate)
    error = E.generate_error(n=code.n, m=trials, seed=seed)
    syndrome = mod2.commute(error, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)

    L = code.logical_opt
    conf = exact_config(len(L), 2**len(L), device=device, dtype=dtype)
    lopt = mod2.confs_to_opt(confs=conf, gs=L)

    '''
    E1 is the error model of generate tensor netwroks, the error rate may not same as E
    '''
    E1 = Errormodel(e_rate=error_rate)
    correct_number = 0
    for j in range(trials):
        cos = torch.zeros(2**len(L))
        elist = []
        for i in range(2**len(L)):
            l = lopt[i]
            e = mod2.opt_prod(pe[j], l)
            elist.append(e.unsqueeze(dim=0))
            tensors, lm = TN.generate_tensors(e, E1.single_p)
            cos[i] = oe.contract(eq, *tensors, optimize='auto')

        indices = torch.argmax(cos, dim=0)
        recover_opt = mod2.opt_prod(pe[j], lopt[indices])
        check_opt = mod2.opt_prod(recover_opt, error[j])
        check = mod2.commute(code.logical_opt, check_opt)

        fail = check.sum()
        #print(fail)
        if fail == 0 :
            correct_number += 1

    logical_error_rate = 1-correct_number/trials
    print(logical_error_rate)
    return logical_error_rate

def Multi_MLD(d, trials, code, error_rate, device, dtype, seed, t=0):

    TN = CodeTN(code, multi=True, device=device, dtype=dtype)
    eq = TN.ein_eq()

    E = Errormodel(e_rate=error_rate)
    error = E.generate_error(n=code.n, m=trials, seed=seed)
    syndrome = mod2.commute(error, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)

    L = code.logical_opt
    conf = exact_config(len(L), 2**len(L), device=device, dtype=dtype)
    lopt = mod2.confs_to_opt(confs=conf, gs=L)
    
    E1 = Errormodel(e_rate=error_rate)
    '''
    E1 is the error model of generate tensor netwroks, the error rate may not same as E
    '''
    if d<=3:
        cos = []
        elist = []
        for i in range(2**len(L)):
            l = lopt[[i]*pe.size(0)]
            e = mod2.opt_prod(pe, l)
            elist.append(e.unsqueeze(dim=0))
            tensors, lm = TN.generate_tensors(e, E1.single_p)
            cos.append(oe.contract(eq, *tensors, optimize='auto').unsqueeze(dim=1))
        cos = torch.cat(cos, dim=1)
    else:
        t=t
        cos = []
        elist = []
        for i in range(2**len(L)):
            l = lopt[[i]*pe.size(0)]
            e = mod2.opt_prod(pe, l)

            pc = []
            for j in range(t):
                tensors, lm = TN.generate_tensors(e[j*int(trials/t):(j+1)*int(trials/t)], E1.single_p)
                pc.append(oe.contract(eq, *tensors, optimize='auto').unsqueeze(dim=1))

            pc = torch.cat(pc, dim=0)
            cos.append(pc)
            print(pc.size())
        cos = torch.cat(cos, dim=1)
        print(cos.size(0))
    indices = torch.argmax(cos, dim=1)
    #print(index)
    #print(cos)
    recover_opt = mod2.opt_prod(pe, lopt[indices])
    #print(recover_opt.size())
    check_opt = mod2.opt_prod(recover_opt, error)

    check = mod2.commute(code.logical_opt, check_opt)

    fail_number = torch.count_nonzero(torch.sum(check, dim=0))
    print(fail_number.item())

    logical_error_rate = fail_number/trials
    print(logical_error_rate.item())
    return logical_error_rate.item()
if __name__ =='__main__':
    import time
    multi = True#False#
    device, dtype = 'cuda:3', torch.float64
    trials = 10000
    c_type='ldpc'
    d, k, seed = 4, 3, 0
    t_p = 1000
    error_seed = 10000
    
    info = read_code(d, k, seed, c_type=c_type)
    code = Loading_code(info)
    mod2 = mod2(device=device, dtype=dtype)

    #error_rate = torch.linspace(0, 0.3, 30)
    error_rate = torch.linspace(0.01, 0.25, 20)#[0.189]#
    #error_rate = torch.linspace(0.01, 0.368, 19)
    lo_rate = []
    t=torch.zeros(len(error_rate))
    for i in range(len(error_rate)):
        t0 = time.time()
        if multi == True:
            lrate = Multi_MLD(d = d, trials=trials, code=code, error_rate=error_rate[i], device=device, dtype=dtype, seed=error_seed, t=t_p)
        else:
            lrate = MLD(d = d, trials=trials, code=code, error_rate=error_rate[i], device=device, dtype=dtype, seed=error_seed)
        lo_rate.append(lrate)
        t1 = time.time()
        t[i]= (t1-t0)/trials
        print((t1-t0)/trials)
    print(t.mean().item(), t.std().item())
    print(lo_rate)
    torch.save((lo_rate, t), abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_exact_{}.pt'.format(d, k, seed, error_seed))
    #torch.save((lo_rate, t), abspath(dirname(__file__))+'/lo_rate/'+c_type+'_d{}_k{}_seed{}_exact_{}_1.pt'.format(d, k, seed, error_seed))