import json
import os
import numpy as np

def main():
    with open('outputs/results.json', 'r') as f:
        results = json.load(f)
        
    dist_data = np.load('outputs/dist_matrices.npz')
    ref = dist_data['float64']
    
    formats = ['float64', 'float32', 'float16', 'int8']
    
    with open('outputs/analysis_summary.txt', 'w') as f:
        f.write("=== Precision Analysis Summary ===\n\n")
        f.write(f"{'Format':<10} | {'Intra':<10} | {'Inter':<10} | {'Ratio':<10} | {'Time (s)':<10} | {'Size (KB)':<10} | {'Mean Abs Error':<15}\n")
        f.write("-" * 85 + "\n")
        
        for name in formats:
            # Memory size
            file_path = f'outputs/rep_{name}.npy'
            size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            
            # Mean abs error
            diff = np.abs(ref - dist_data[name])
            mean_error = diff.mean()
            
            intra = results[name]['intra']
            inter = results[name]['inter']
            ratio = results[name]['ratio']
            time_s = results[name]['time_s']
            
            f.write(f"{name:<10} | {intra:<10.4f} | {inter:<10.4f} | {ratio:<10.4f} | {time_s:<10.4f} | {size_kb:<10.2f} | {mean_error:<15.6f}\n")

if __name__ == '__main__':
    main()
