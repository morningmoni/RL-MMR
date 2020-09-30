import numpy as np


def calc_sent_pos(sent_l):
    n_bins = 6
    feat = np.zeros((len(sent_l), n_bins))
    for i in range(n_bins - 1):
        feat[i, i] = 1
    feat[n_bins - 1:, n_bins - 1] = 1
    return feat


def calc_sent_len(sent_l):
    MAX_LEN = 100
    n_bins = 5
    interval = MAX_LEN / n_bins
    feat = np.zeros((len(sent_l), n_bins))
    for i, sent in enumerate(sent_l):
        bin_idx = int(len(sent) / interval)
        feat[i, min(bin_idx, n_bins - 1)] = 1
    return feat


def calc_feat(sent_l):
    feat = np.concatenate([calc_sent_pos(sent_l), calc_sent_len(sent_l)], axis=1)
    return feat
