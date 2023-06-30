from args import args
import torch
import time
import sys
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Errormodel, mod2, Loading_code, read_code
torch.backends.cudnn.enable =True

c_type, d, k, seed = 'sur', 3, 7, 0
'''Net Paras:'''
l_type, n_type, e_model = 'fkl', args.n_type, args.e_model
er = 0.15
'''seed for sampling errors:'''
error_seed = args.error_seed
path = abspath(dirname(__file__))+'/net/'+l_type+'_'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_mid.pt'.format(d, k, seed, er)
net = torch.load(path)
print('D:', net.d_model)
print('N_h:', net.n_heads)
print('N_l:', net.n_layers)
print('D_f:', net.d_ff)