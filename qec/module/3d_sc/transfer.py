import numpy as np
import torch

for d in range(2, 10):
    code_file = f'3d_sc/3d_sc_d{d}'
    pcm_file = code_file + '_pcm.npz'
    logicals_file = code_file + '_logicals.npz'
    H = np.load(pcm_file)['arr_0']
    tmp = np.load(logicals_file)['arr_0']
    m = d ** 3 - d ** 2
    Hx = H[:m]
    Hz = H[m:]

    logical_operators = np.zeros([4, tmp.shape[1], tmp.shape[2]])
    logical_operators[1:3] = tmp[:, ::-1]
    logical_operators[3] = (logical_operators[1] + logical_operators[2]) % 2
    print(d, Hx.shape, Hz.shape, logical_operators.shape)
    torch.save(
        (torch.from_numpy(Hx), torch.from_numpy(Hz), torch.from_numpy(logical_operators)),
        f'3d_sc/3d_sc_d{d}.pt'
    )