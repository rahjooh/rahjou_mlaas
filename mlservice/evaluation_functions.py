"""
EVALUATION FUNCTIONS
"""

import sys

import numpy as np
import sklearn.metrics

sys.path.append('/root/jupyter/')
from mlservice.common import util


# from importlib import reload
# util=reload(util)


def mse(Ytrue, Ypred):
    if len(Ypred.shape) == 1:
        Ypred = util.oneofk(Ypred)
    if len(Ytrue.shape) == 1:
        Ytrue = util.oneofk(Ytrue)
    return ((Ypred - Ytrue) ** 2).mean()


def auc(Ytrue, Ypred):
    if len(Ytrue.shape) != 1:
        Ytrue = util.oneofk_reverse(Ytrue)
    assert (len(Ypred.shape) <= 2)
    if len(Ypred.shape) == 2:
        Ypred = Ypred[:, 1]

    return sklearn.metrics.roc_auc_score(Ytrue, Ypred)


def confusion(Ytrue, Ypred):
    if len(Ytrue.shape) != 1:
        Ytrue = util.oneofk_reverse(Ytrue)
    if len(Ypred.shape) != 1:
        Ypred = util.oneofk_reverse(Ypred)

    return sklearn.metrics.confusion_matrix(Ytrue, Ypred)


#     C=max(Ytrue.max(), Ypred.max())+1
#     confusion_count=np.zeros((C,C))

#     cython_inc_at2d(confusion_count, Ytrue, Ypred, 1.0)
#     return confusion_count/confusion_count.sum(axis=1, keepdims=True)


def accuracy(Ytrue, Ypred):
    if len(Ytrue.shape) != 1:
        Ytrue = util.oneofk_reverse(Ytrue)
    if len(Ypred.shape) != 1:
        Ypred = util.oneofk_reverse(Ypred)

    return (Ytrue == Ypred).mean()


def logistic_loss(Ytrue, Ypred):
    if len(Ytrue.shape) != 1:
        Ytrue = util.oneofk_reverse(Ytrue)
    assert (len(Ypred.shape) <= 2)
    if len(Ypred.shape) == 2:
        Ypred = Ypred[:, 1]

    return sklearn.metrics.log_loss(Ytrue, Ypred)


def outbrain_mean_precision_atK(Ytrue, Ypred, ranking_group, K=12, goal_class=1, return_extra_info=True):
    if len(Ypred.shape) != 1:
        Ypred = Ypred[:, goal_class]

    Ytrue, Ypred, ranking_group = [elem.copy() for elem in [Ytrue, Ypred, ranking_group]]

    # # We first shuffle data to make sure they are not in paticular order based on Ytrue...=> Replaced by sorting Ytrue
    # randIndcs = np.random.permutation(np.arange(Ytrue.shape[0]))
    # Ytrue, Ypred, ranking_group = [elem[randIndcs] for elem in [Ytrue, Ypred, ranking_group]]

    sortIndcs = np.lexsort([Ytrue == goal_class, -Ypred, ranking_group])
    Ytrue, Ypred, ranking_group = [elem[sortIndcs] for elem in [Ytrue, Ypred, ranking_group]]

    uniqVals, uniqCounts = np.unique(ranking_group, return_counts=True)
    uniqVals, uniqCounts = np.unique(uniqCounts, return_counts=True)
    extra_info = {'ranking_groups_size_frequency': {size: freq for size, freq in zip(list(uniqVals), list(uniqCounts))}}

    assert ((Ytrue == 1) | (Ytrue == 0)).sum() == Ytrue.shape[0], 'Ytrue should be binary (only 0 and 1 values)'

    s = 0
    i = 0
    N = 0
    while i < len(Ytrue):
        nexti = i + 1
        cur_group = ranking_group[i]
        ni = 1
        while nexti < len(Ytrue) and (cur_group == ranking_group[nexti]):
            nexti += 1
            ni += 1

        cury = Ytrue[i:i + min(ni, K)]

        for k in range(1, len(cury) + 1):
            s += cury[:k].sum() / k

        N += 1
        i = nexti

    s /= N

    if return_extra_info:
        return s, extra_info
    return s


def mean_precision_atK(Ytrue, Ypred, ranking_group, ad_ids, K=12, goal_class=1, return_extra_info=True):
    # Based on equation 8.8 of "Schütze, Hinrich, Christopher D. Manning, and Prabhakar Raghavan. "Introduction to information retrieval." Proceedings of the international communication of association for computing machinery conference. 2008."
    #     print('tmp')
    if len(Ypred.shape) != 1:
        Ypred = Ypred[:, goal_class]

    Ytrue, Ypred, ranking_group = [elem.copy() for elem in [Ytrue, Ypred, ranking_group]]
    Ytrue = Ytrue == goal_class

    # # We first shuffle data to make sure they are not in paticular order based on Ytrue...=> Replaced by sorting Ytrue
    # randIndcs = np.random.permutation(np.arange(Ytrue.shape[0]))
    # Ytrue, Ypred, ranking_group = [elem[randIndcs] for elem in [Ytrue, Ypred, ranking_group]]

    sortIndcs = np.lexsort([Ytrue, -Ypred.astype(np.float), ranking_group])
    Ytrue, Ypred, ranking_group = [elem[sortIndcs] for elem in [Ytrue, Ypred, ranking_group]]

    uniqVals, uniqCounts = np.unique(ranking_group, return_counts=True)
    uniqVals, uniqCounts = np.unique(uniqCounts, return_counts=True)
    extra_info = {'ranking_groups_size_frequency': {size: freq for size, freq in zip(list(uniqVals), list(uniqCounts))},
                  'Ytrue': Ytrue, 'Ypred': Ypred, 'ranking_group': ranking_group
                  }

    assert ((Ytrue == 1) | (Ytrue == 0)).sum() == Ytrue.shape[0], 'Ytrue should be binary (only 0 and 1 values)'

    s = 0
    i = 0
    N = 0
    tmp_summ = 0
    while i < len(Ytrue):
        nexti = i + 1
        cur_group = ranking_group[i]
        ni = 1

        while nexti < len(Ytrue) and (cur_group == ranking_group[nexti]):
            nexti += 1
            ni += 1

        # For the cases where the ad_id is repeated, the first occurance is used:
        clicked_ads = set()
        for j in range(i, i + min(ni, K)):
            if Ytrue[j]:
                clicked_ads.add(ad_ids[j])

        for j in range(i, i + min(ni, K)):
            if ad_ids[j] in clicked_ads:
                Ytrue[j] = True
                clicked_ads.remove(ad_ids[j])
            else:
                Ytrue[j] = False

        # Calculate MAP@K for current ranking instance
        cury = Ytrue[i:i + min(ni, K)]
        m = cury.sum()
        tmp_summ += m

        if (m > 1):
            print('m>1: ', m)

        ranks_of_clicked: np.ndarray = np.where(cury)[0] + 1
        if ranks_of_clicked.size != 0:
            s += (1 / ranks_of_clicked) / m

        N += 1
        i = nexti

    s /= N
    # print(tmp_summ / N)
    if return_extra_info:
        return s, extra_info
    return s
