import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from datasets.paths import CAT_DATASET_PATH
from datasets.paths import MOUSEACH_DATASET_PATH
from datasets.paths import MOUSE_DATASET_PATH
from datasets.LFPDataset import LFPDataset


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


def load_dataset(path):
    dataset = LFPDataset(path)
    mask = np.ones(len(dataset.channels), dtype=bool)
    if path == MOUSE_DATASET_PATH:
        mask[[3, 6, 32]] = False
        dataset.nr_channels -= 3
    elif path == MOUSEACH_DATASET_PATH:
        mask[[1, 3, 6, 30, 32]] = False
        dataset.nr_channels -= 5
    elif path == CAT_DATASET_PATH:
        pass
    dataset.channels = dataset.channels[mask]

    return dataset


def heatmap_from(data,
                 distance_metric,
                 save_location,
                 plot_title,
                 methods=["average"],
                 add_order_labels=True):
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    dist_mat = squareform(pdist(data, distance_metric))
    N = len(data)

    # plt.pcolormesh(dist_mat)
    # plt.xlim([0, N])
    # plt.ylim([0, N])
    # plt.savefig(os.path.join("./heatmaps/{0}-Method:{1}.png".format(plot_title, "unordered")))

    results = []
    for method in methods:
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

        plt.pcolormesh(ordered_dist_mat)
        plt.xlim([0, N])
        plt.ylim([0, N])
        if add_order_labels:
            plt.yticks([x for x in range(len(res_order))], res_order)
        plt.savefig(os.path.join(save_location, "{0}-{1}.png".format(plot_title, method)))
        # plt.show()

        results.append((ordered_dist_mat, res_order, res_linkage, dist_mat))
    return results


def compute_heatmap_on_dataset(path):
    output_path = "./heatmaps"
    plot_title = ""

    if path == MOUSE_DATASET_PATH:
        plot_title = "Mouse LFP Channels"
    elif path == MOUSEACH_DATASET_PATH:
        plot_title = "Mouse ACh LFP Channels"
    elif path == CAT_DATASET_PATH:
        plot_title = "Cat LFP Channels"

    dataset = load_dataset(path)

    heatmap_from(
        dataset.channels,
        'correlation',
        output_path,
        plot_title,
        add_order_labels=True)


def compute_heatmaps_on_channels():
    compute_heatmap_on_dataset(CAT_DATASET_PATH)
    compute_heatmap_on_dataset(MOUSE_DATASET_PATH)
    compute_heatmap_on_dataset(MOUSEACH_DATASET_PATH)


def heatmaps_cross_channels_on_dataset(path):
    dataset = load_dataset(path)

    output_path = "./heatmaps/cross_channels"
    if path == MOUSE_DATASET_PATH:
        output_path += "/mouse"
    elif path == MOUSEACH_DATASET_PATH:
        output_path += "/mouse_ach"
    elif path == CAT_DATASET_PATH:
        output_path += "/cat"

    for stimulus_condition in dataset.stimulus_conditions:
        index = int(stimulus_condition['Trial']) - 1
        events = [{'timestamp': dataset.event_timestamps[4 * index + i],
                   'code': dataset.event_codes[4 * index + i]} for i in range(4)]
        trial = dataset.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)]
        heatmap_from(trial,
                     'correlation',
                     output_path,
                     "Trial {}-Condition number{}-{}-{}".format(
                         stimulus_condition['Trial'],
                         stimulus_condition['Condition number'],
                         stimulus_condition['Condition name'],
                         stimulus_condition['Duration (us)']))


def compute_heatmaps_cross_channels():
    heatmaps_cross_channels_on_dataset(MOUSE_DATASET_PATH)
    heatmaps_cross_channels_on_dataset(MOUSEACH_DATASET_PATH)


def heatmap_cross_trials(path):
    dataset = load_dataset(path)

    output_path = "./heatmaps/cross_trials"
    if path == MOUSE_DATASET_PATH:
        output_path += "/mouse"
    elif path == MOUSEACH_DATASET_PATH:
        output_path += "/mouse_ach"
    elif path == CAT_DATASET_PATH:
        output_path += "/cat"

    cond_trial_channel = []
    for condition in range(1, dataset.number_of_conditions + 1):
        conditions = []
        for stimulus_condition in dataset.stimulus_conditions:
            if stimulus_condition['Condition number'] == str(condition):
                index = int(stimulus_condition['Trial']) - 1
                events = [{'timestamp': dataset.event_timestamps[4 * index + i],
                           'code': dataset.event_codes[4 * index + i]} for i in range(4)]
                # trial = self.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)]
                # Right now it cuts only the area where the stimulus is active
                # In order to keep the whole trial replace with
                trial = dataset.channels[:, events[0]['timestamp']:(events[0]['timestamp'] + 4175)]
                conditions.append(trial)
        cond_trial_channel.append(np.array(conditions))
    cond_trial_channel = np.array(cond_trial_channel, dtype=np.float32)

    for condition in range(0, dataset.number_of_conditions):
        for channel in range(0, dataset.nr_channels):
            heatmap_from(cond_trial_channel[condition, :, channel, :],
                         'correlation',
                         output_path,
                         "Condition number{}-Channel{}".format(condition, channel))


def compute_heatmaps_cross_trials():
    heatmap_cross_trials(MOUSE_DATASET_PATH)
    heatmap_cross_trials(MOUSEACH_DATASET_PATH)


if __name__ == '__main__':
    compute_heatmaps_on_channels()
    compute_heatmaps_cross_channels()
    compute_heatmaps_cross_trials()
