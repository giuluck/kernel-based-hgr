import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.datasets import Communities
from src.hgr import KernelBasedHGR

if __name__ == '__main__':
    # set graphics context
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    # retrieve data
    a = dataset.excluded(backend='numpy')
    b = dataset.target(backend='numpy')
    # build results matrix and iterate through degrees
    results = np.zeros((len(degrees_a), len(degrees_b)), dtype=float)
    pbar = tqdm(total=len(degrees_a) * len(degrees_b))
    for i, da in enumerate(degrees_a):
        for j, db in enumerate(degrees_b):
            metric = KernelBasedHGR(degree_a=da, degree_b=db)
            # compute kernel matrices and retrieve the hgr value
            f = np.stack([a ** d - np.mean(a ** d) for d in np.arange(da) + 1], axis=1)
            g = np.stack([b ** d - np.mean(b ** d) for d in np.arange(db) + 1], axis=1)
            hgr, alp, bet = metric.kbhgr(a=a, b=b)
            results[i, j] = hgr
            # plot kernels (if necessary)
            if kernels:
                fig, axes = plt.subplots(2, 2, figsize=(16, 9), tight_layout=True)
                fa, gb = f @ alp, g @ bet
                sns.scatterplot(x=a, y=b, size=20, alpha=0.4, color='black', legend=None, ax=axes[0, 0])
                axes[0, 0].set_title('Original Data')
                sns.scatterplot(x=b, y=gb, size=20, alpha=1, color='black', legend=None, ax=axes[0, 1])
                axes[0, 1].set_title('Y Kernel')
                sns.scatterplot(x=a, y=fa, size=20, alpha=1, color='black', legend=None, ax=axes[1, 0])
                axes[1, 0].set_title('X Kernel')
                sns.scatterplot(x=fa, y=gb, size=20, alpha=0.4, color='black', legend=None, ax=axes[1, 1])
                axes[1, 1].set_title('Projected Data')
                fig.suptitle(f'Kernels for {dataset.name.title()} (dx = {da}, dy = {db}) --> HGR = {hgr:0.4}')
                fig.show()
            # update progress bar
            pbar.update(n=1)
    pbar.close()
    # plot results
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = fig.gca()
    col = ax.imshow(results.transpose()[::-1], cmap=plt.colormaps['gray'], vmin=vmin, vmax=vmax)
    fig.colorbar(col, ax=ax)
    ax.set_xlabel('Degree A')
    ax.set_xticks(np.arange(len(degrees_a) + 1) - 0.5)
    ax.set_xticklabels([''] * (len(degrees_a) + 1))
    ax.set_xticks(np.arange(len(degrees_a)), minor=True)
    ax.set_xticklabels(degrees_a, minor=True)
    ax.set_ylabel('Degree B')
    ax.set_yticks(np.arange(len(degrees_b) + 1) - 0.5)
    ax.set_yticklabels([''] * (len(degrees_b) + 1))
    ax.set_yticks(np.arange(len(degrees_b)), minor=True)
    ax.set_yticklabels(degrees_b[::-1], minor=True)
    ax.grid(True, which='major')
    fig.show()
