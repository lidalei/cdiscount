import json, pickle, argparse
import matplotlib.pylab as plt
from os.path import join as path_join
import numpy as np

folder = '.'

with open(folder + '/sum_labels_one_hot.pickle', 'rb') as pickle_file:
    sum_labels = pickle.load(pickle_file)

    sum_labels_with_id = enumerate(sum_labels)

    sorted_sum_labels = sorted(sum_labels_with_id, key=lambda x: x[1], reverse=True)

    sorted_idx, sorted_labels = zip(*sorted_sum_labels)

    figure, ax = plt.subplots(1)
    ax.semilogy(sorted_labels, color='#1f77b4', marker='.', markersize=0.1, label='Number of instances')
    ax.legend(loc=1)
    ax.set_xlabel('Labels')
    ax.set_xticklabels([str(id) for id in sorted_idx], rotation='vertical')
    ax.tick_params('y', colors='#1f77b4')

    ax.grid()

    plt.show()
