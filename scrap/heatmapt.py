import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    Z = np.random.random((20, 100))
    fig, ax = plt.subplots(figsize=(5, 2))
    ax = sns.heatmap(Z, ax=ax, cmap="Greys", linewidths=0.0, xticklabels=False, yticklabels=False)
    ax.set_ylabel('{}'.format('COMPLEXITY'))
    ax.invert_yaxis()
    # ax.set_ylim(bounds[0], bounds[1])
    ax.set_xlabel('Timestep')
    plt.yticks([0, 20], [0.5, 4.5])
    plt.xticks([0, 100], [0, 200])
    #ax.tick_params(axis='both', left=True, right=True)
    #ax.tick_params(axis='x', left=True, right=True)
    #ax.tick_params(direction='out', length=6, width=2, colors='r')
    fig.tight_layout()
    #fig_name = 'heatmap_{}_{}_r{:0>4}.pdf'.format(model, aest, run_id + 1)
    #fig_path = os.path.join(save_folder, fig_name)
    #plt.imshow(extent=[1, 100, 0.5, 4.5])
    plt.savefig('aa_test.pdf')
    plt.show()
