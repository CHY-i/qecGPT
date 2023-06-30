from args import args
import numpy as np
import torch
import time
import sys
from ldpc import bposd_decoder
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, Hx_Hz, Errormodel, mod2

d, k, seed, c_type = args.d, args.k, args.seed, args.c_type
trials = args.trials
device, dtype = 'cpu', torch.float64
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
print(l)

#error_rate = torch.linspace(0.01, 0.25, 20)#[0.01]#
if k == 1 :
    error_rate = torch.linspace(0.01, 0.368, 19)
elif k==2 and c_type=='tor':
    error_rate = torch.linspace(0.01, 0.368, 19)
else:
    error_rate = torch.linspace(0.01, 0.25, 20)
L = []
tt = torch.zeros(len(error_rate))
for i in range(len(error_rate)):
    E = Errormodel(e_rate=error_rate[i])

    bpd=bposd_decoder(
    PCM,#the parity check matrix
    error_rate=2*error_rate[i],
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=1000, #the maximum number of iterations for BP)
    bp_method="ms",
    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=7 #the osd search depth
    )

    error = E.generate_error(n=code.n, m=trials, seed=0)
    syndrome = mod2.commute(error, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)
    error = mod2.rep(error).squeeze().int().numpy()
    syndrome = syndrome.numpy()

    correct_number = 0
    t = 0
    for j in range(trials):
        e = error[j]
        s = syndrome[j]

        t1 = time.time()
        bpd.decode(s)
        recover = bpd.osdw_decoding
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
        # print('Error:')
        # print(e)
        # print('Decoding:')
        # print(recover)
        # print('Check:')
        # print(check)
