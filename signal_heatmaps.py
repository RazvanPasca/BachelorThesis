import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from datasets.DATASET_PATHS import GABI_CAT_DATASET_PATH as CAT_DATASET_PATH
from datasets.DATASET_PATHS import GABI_MOUSE_DATASET_PATH as MOUSE_DATASET_PATH
from datasets.DATASET_PATHS import GABI_MOUSEACH_DATASET_PATH as MOUSEACH_DATASET_PATH
from datasets.LFPDataset import LFPDataset


def compute_distance_matrix(dataset, distance_metric):
    dist_mat = squareform(pdist(dataset.channels, distance_metric))
    N = len(dataset.channels)
    methods = ["ward", "single", "average", "complete"]
    plt.pcolormesh(dist_mat)
    plt.xlim([0, N])
    plt.ylim([0, N])
    plt.show()
    for method in methods:
        print("Method:\t", method)

        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

        plt.pcolormesh(ordered_dist_mat)
        plt.xlim([0, N])
        plt.ylim([0, N])
        plt.show()


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


if __name__ == '__main__':
    dataset = LFPDataset(MOUSEACH_DATASET_PATH)
    dataset.channels = dataset.channels[:-1]
    dataset.nr_channels -=1
    compute_distance_matrix(dataset, 'correlation')

    dataset = LFPDataset(MOUSE_DATASET_PATH)
    dataset.channels = dataset.channels[:-1]
    dataset.nr_channels -=1
    compute_distance_matrix(dataset, 'correlation')


