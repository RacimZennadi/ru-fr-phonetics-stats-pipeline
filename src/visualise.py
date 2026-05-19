import numpy as np
import matplotlib.pyplot as plt
import os
import json

with open('outputs/labels.json', 'r') as f:
    labels = json.load(f)

dist_data = np.load('outputs/dist_matrices.npz')
formats = ['float64', 'float32', 'float16', 'int8']

os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, name in enumerate(formats):
    ax = axes[idx]
    dists = dist_data[name]
    
    intra = []
    inter = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i]['word'] == labels[j]['word']:
                if labels[i]['speaker'] == labels[j]['speaker']:
                    intra.append(dists[i, j])
                else:
                    inter.append(dists[i, j])
                    
    ax.hist(intra, bins=30, range=(0.0, 0.5), alpha=0.6, label='Intra-speaker')
    ax.hist(inter, bins=30, range=(0.0, 0.5), alpha=0.6, label='Inter-speaker')
    ax.set_title(f'{name}')
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('figures/distance_distributions.png')

with open('outputs/results.json', 'r') as f:
    results = json.load(f)

intra_means = [results[f]['intra'] for f in formats]
inter_means = [results[f]['inter'] for f in formats]
ratios = [results[f]['ratio'] for f in formats]

x = np.arange(len(formats))
width = 0.3
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, intra_means, width, label='Intra-speaker')
ax.bar(x + width/2, inter_means, width, label='Inter-speaker')
ax.set_xticks(x)
ax.set_xticklabels(formats)
ax.set_ylabel('Mean cosine distance')
ax.set_title('Intra vs Inter-speaker distances by precision')
ax.legend()
plt.tight_layout()
plt.savefig('figures/intra_inter_comparison.png')
