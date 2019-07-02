import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from datasets import CatDataset, MouseControlDataset, MouseAChDataset


# methods = ["ward", "single", "average", "complete"]

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


def heatmap_from(data,
                 distance_metric,
                 save_location,
                 plot_title,
                 methods=["average"],
                 add_order_labels=True,
                 show=False):
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    dist_mat = squareform(pdist(data, distance_metric))
    N = len(data)

    # plt.pcolormesh(dist_mat)
    # plt.xlim([0, N])
    # plt.ylim([0, N])
    # plt.savefig(os.path.join("./heatmapss/{0}-Method:{1}.png".format(plot_title, "unordered")))

    results = []
    for method in methods:
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
        plt.figure(figsize=(20, 10))
        plt.pcolormesh(ordered_dist_mat)
        plt.xlim([0, N])
        plt.ylim([0, N])
        plt.xlabel("Channels")
        plt.ylabel("Channels")
        plt.colorbar()
        if add_order_labels:
            plt.yticks([x for x in range(len(res_order))], res_order)
            plt.xticks([x for x in range(len(res_order))], res_order)
        plt.savefig(os.path.join(save_location, "{0}-{1}.png".format(plot_title, method)))
        if show:
            plt.show()
        plt.close()
        results.append((ordered_dist_mat, res_order, res_linkage, dist_mat))
    return results


def compute_heatmaps_cross_channels(dataset):
    output_path = "./heatmaps/cross_channels/" + dataset.get_name()

    for condition in range(dataset.number_of_conditions):
        for trial in range(dataset.trials_per_condition):
            trial_signal = dataset.signal[condition, trial]
            heatmap_from(trial_signal,
                         'correlation',
                         output_path,
                         "Trial {}-Condition number {}".format(
                             trial,
                             condition))


def compute_heatmaps_cross_trials(dataset):
    output_path = "./heatmaps/cross_trials/" + dataset.get_name()

    for condition in range(0, dataset.number_of_conditions):
        for channel in range(0, dataset.number_of_channels):
            heatmap_from(dataset.signal[condition, :, channel, :],
                         'correlation',
                         output_path,
                         "Condition number{}-Channel{}".format(condition, channel))


if __name__ == '__main__':
    # compute_heatmaps_cross_channels(CatDataset())
    compute_heatmaps_cross_channels(MouseControlDataset())
    compute_heatmaps_cross_channels(MouseAChDataset())
    # compute_heatmaps_cross_trials(CatDataset())
    compute_heatmaps_cross_trials(MouseControlDataset())
    compute_heatmaps_cross_trials(MouseAChDataset())
