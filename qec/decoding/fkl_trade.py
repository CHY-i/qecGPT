
from args import args
import torch
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import TraDE, Errormodel, mod2, Loading_code, read_code, btype

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type =='made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type == 'trade':
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]
    return x

if __name__ == '__main__':
    from module import Surfacecode
    n_type = 'trade'
    save = args.save
    dtype=torch.float64
    trials = args.trials
    d, k, device, seed = args.d, args.k, args.device, args.seed
    epoch, batch, lr = args.epoch, args.batch, args.lr
    
    c_type = args.c_type
    seq = args.seq
    e_model = args.e_model


    mod2 = mod2(device=device, dtype=dtype)
    info = read_code(d, k, seed, c_type=c_type)
    Code = Loading_code(info)
    
    e1 = Code.pure_es 
    for i in range(Code.m):
        conf = mod2.commute(e1[i], e1)
        idx = conf.nonzero().squeeze()
        sta = Code.g_stabilizer[idx]
        e1[i] = mod2.opts_prod(torch.vstack([e1[i], sta]))

    if seq == 'mid':
        g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer])##
    elif seq == 'fin':
        g = torch.vstack([e1, Code.g_stabilizer, Code.logical_opt])


    if e_model == 'depolarized':
        er = args.er
        E = Errormodel(er, e_model=e_model)
        errors = E.generate_error(Code.n,  m=trials, seed=seed)
    elif e_model == 'ising':
        beta = args.beta
        h = args.h
        E = Errormodel()
        errors, _ = E.ising_error(n_s=trials, n=Code.n, beta=beta, h=h)
    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)
    
    kwargs_dict = {
        'n': g.size(0),
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        'device': device,#'cuda:5'
        'dropout': 0, 
    }

    

    van = TraDE(**kwargs_dict).to(kwargs_dict['device']).to(dtype)

    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    # loss_his = []
    # lo_his = []
    for l in range(epoch):
        if e_model == 'depolarized':
            ers = E.generate_error(Code.n, m=batch, seed=False)
        elif e_model == 'ising':
            ers,_ = E.ising_error(n_s=batch, n=Code.n , beta=beta, h=h)
        configs = E.configs(sta=Code.g_stabilizer, log=Code.logical_opt, pe=e1, opts=ers, seq=seq).to(device)

        logp = van.log_prob(configs)
       
        loss = torch.mean((-logp), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
           scheduler.step()
        if (l+1)%100==0 and e_model != 'ising':
            logq = E.log_probability(opts=ers, device=device, dtype=dtype)
            KL = torch.mean(logq-logp, dim=0)
            print(l, KL)
            # loss_his.append(KL)
        if (l+1) % args.cpe == 0:
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
            # lo_his.append(logical_error_rate)
    # print(loss_his)
    # print(lo_his)
    # torch.save((loss_his, lo_his), abspath(dirname(__file__))+'/his.pt')
    if save == True:
        if e_model == 'depolarized':
            path = abspath(dirname(__file__))+'/net/fkl_trade_'+c_type+'_d{}_k{}_seed{}_er{}_{}.pt'.format(d, k, seed, er, seq)
            if exists(path):
                None
            else:          
                torch.save(van, path)
        elif e_model == 'ising':
            path = abspath(dirname(__file__))+'/net/fkl_trade_'+c_type+'_d{}_k{}_seed{}_ising{}_h{}_{}.pt'.format(d, k, seed, beta, h, seq)
            if exists(path):
                None
            else:  
                torch.save(van, path)  
