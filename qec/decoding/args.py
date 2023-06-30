import argparse

parser = argparse.ArgumentParser()

par_common = parser.add_argument_group('common parameters')
'''para of code'''
par_common.add_argument('-c_type', type=str, default='sur',
        help='the code type of the original code, one of the labels of code, default: %(default)s')
par_common.add_argument('--d', type=int, default=3,
        help='the distance of the original code, one of the labels of code')
par_common.add_argument('--k', type=int, default=1,
        help='the number of logical qubits of the code, one of the labels of code, default: %(default)d')
par_common.add_argument('--seed', type=int, default=0,
        help='seed of random removal of stabilizers from the original code, one of the labels of code, default: %(default)d')
'''para of errors'''
par_common.add_argument('-e_model', type=str, default='depolarized',
        help='error model, default: %(default)d')
par_common.add_argument('--error_seed', type=int, default=10000,
        help='seed of generate errors, default: %(default)d')
par_common.add_argument('--trials', type=int, default=10000,
        help='trials of decoding, default: %(default)d')
par_common.add_argument('--er', type=float, default=0.189,
        help='the error rate for inference, default: %(default)d')
par_common.add_argument('--beta', type=float, default=0.3,
        help='the inverse temperature of ising error model, default: %(default)d')
par_common.add_argument('--h', type=float, default=1.0,
        help='the external field of ising error model, default: %(default)d')

'''para of made'''
par_common.add_argument('--depth', type=int, default=0,
        help='depth of MADE, default: %(default)d')
par_common.add_argument('--width', type=int, default=1,
        help='width of MADE, default: %(default)d')

'''para of trade'''
par_common.add_argument('--d_model', type=int, default=256,
        help='d_model of trade, default: %(default)d')
par_common.add_argument('--n_heads', type=int, default=4,
        help='number of heads, default: %(default)d')
par_common.add_argument('--d_ff', type=int, default=256,
        help='dim of forward, default: %(default)d')
par_common.add_argument('--n_layers', type=int, default=1,
        help='number of layers, default: %(default)d')
'''para for training'''
par_common.add_argument('-n_type', type=str, default='trade', choices=['made', 'trade'],
        help='net type of training , default: %(default)s')
par_common.add_argument('-seq', type=str, default='mid',
        help='position of logical variables , default: %(default)s')

par_common.add_argument('-dtype', type=str, default='float32',
        choices=['float32', 'float64'],
        help='dtypes used during training, default: %(default)s')
par_common.add_argument('-device', type=str, default='cuda:0',
        help='device used during training, default: %(default)s')
par_common.add_argument('--epoch', type=int, default=10000,
        help='epoch of training, default: %(default)d')
par_common.add_argument('--batch', type=int, default=10000,
        help='batch of training, default: %(default)d')
par_common.add_argument('--lr', type=float, default=0.001,
        help='learning rate, default: %(default)d')
par_common.add_argument('--cpe', type=int, default=10000,
        help='correction per cpe epoch, default: %(default)s')

par_common.add_argument('-save', type=bool, default=False,
        help='save the results if true, default: %(default)s')
        

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
