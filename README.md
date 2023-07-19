# GenerativeDecoder
Training a autoregressive network to do quantum error correction.

Link to the article: https://doi.org/10.48550/arXiv.2307.09025

## Generate Surface Code d=3 k=1:

```python
python code_generator.py --d 3 --k 1 --seed 0 -c_type 'sur'
```
if k> k' , where k' is the number of logical qubits, will remove stabilizers randomly from origin code.

## Training a MADE or TraDE and save the network

### MADE
```python
python fkl_made.py -save True -c_type 'sur' --d 3 --k 1 --seed 0 -device 'cuda:0' --batch 10000 --epoch 50000 --depth 3 --width 10
```
### TraDE
```python
python fkl_trade.py -save True -c_type 'sur' --d 3 --k 1 --seed 0 -device 'cuda:0' --batch 10000 --epoch 50000 --n_layers 2
```
## Correction

### Loading network and forward to do error correction and save the logical error rate

### depolarized
```python
python forward_mid.py -save True -c_type 'sur' --d 3 --k 1 --seed 0  -device 'cuda:0' -n_type 'made'/'trade' -e_model 'depolarized' --trials 10000 --error_seed 10000 --er 0.189
```
### ising
```python
python forward_mid.py -save True -c_type 'sur' --d 3 --k 1 --seed 0  -device 'cuda:0' -n_type 'made' -e_model 'ising' --trials 10000 --beta 0.3 --h 1.0
```

