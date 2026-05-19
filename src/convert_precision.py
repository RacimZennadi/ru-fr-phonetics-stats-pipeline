import numpy as np
import os
import json

os.makedirs('outputs', exist_ok=True)

def quantize_8bit(x):
    x_min, x_max = x.min(), x.max()
    scale = float((x_max - x_min) / 255.0)
    x_min_f = float(x_min)
    x_q = np.round((x - x_min_f) / scale)
    return x_q.astype(np.uint8), scale, x_min_f

data_64 = np.load('outputs/rep_float64.npy')

data_32 = data_64.astype(np.float32)
data_16 = data_64.astype(np.float16)
data_8q, scale_8, min_8 = quantize_8bit(data_64)

np.save('outputs/rep_float32.npy', data_32)
np.save('outputs/rep_float16.npy', data_16)
np.save('outputs/rep_int8.npy', data_8q)

with open('outputs/quant_params.json', 'w') as f:
    json.dump({'scale_8': scale_8, 'min_8': min_8}, f)
