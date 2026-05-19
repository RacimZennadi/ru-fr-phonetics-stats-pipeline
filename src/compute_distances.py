import numpy as np
import os
import json
import time

def dequantize_8bit(x_q, scale, x_min):
    return x_q.astype(np.float32) * scale + x_min

def compute_distances(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norms + 1e-10)
    cos_sim = np.dot(normalized, normalized.T)
    return 1.0 - cos_sim

with open('outputs/labels.json', 'r') as f:
    labels = json.load(f)

with open('outputs/quant_params.json', 'r') as f:
    quant_params = json.load(f)

formats = ['float64', 'float32', 'float16', 'int8']
arrays = {fmt: np.load(f'outputs/rep_{fmt}.npy') for fmt in formats}

dist_matrices = {}
time_taken = {}

for name, arr in arrays.items():
    start = time.time()
    
    # dequantize before computing distances
    if name == 'int8':
        arr_calc = dequantize_8bit(arr, quant_params['scale_8'], quant_params['min_8'])
    else:
        arr_calc = arr
        
    dist_matrices[name] = compute_distances(arr_calc)
    time_taken[name] = round(time.time() - start, 6)

results = {}

for name, dists in dist_matrices.items():
    intra = []
    inter = []
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i]['word'] == labels[j]['word']:
                if labels[i]['speaker'] == labels[j]['speaker']:
                    intra.append(dists[i, j])
                else:
                    inter.append(dists[i, j])
                    
    if len(intra) == 0:
        print(f"WARNING: No intra-speaker pairs found for format {name}! The dataset might not contain repeated recordings of the same word by the same speaker.")
        intra_mean = 0.0
    else:
        intra_mean = float(np.mean(intra))

    if len(inter) == 0:
        print(f"WARNING: No inter-speaker pairs found for format {name}!")
        inter_mean = 0.0
    else:
        inter_mean = float(np.mean(inter))
    ratio = float(inter_mean / intra_mean) if intra_mean != 0 else 0.0
    
    results[name] = {
        'intra': intra_mean,
        'inter': inter_mean,
        'ratio': ratio,
        'time_s': time_taken[name]
    }

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=4)

np.savez_compressed('outputs/dist_matrices.npz', **dist_matrices)
