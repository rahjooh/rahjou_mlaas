import sys

sys.path.append('/root/jupyter-base/')
from mlservice.common import util
# from importlib import reload
# util=reload(util)

import numpy as np


class NaiveBayes():

    def __init__(self):
        pass

    def fit(self, X, Y, alpha=1):
        C = Y.max() + 1

        py = np.zeros(C)
        for y in range(C):
            py[y] = (Y == y).mean()

        log_pXi_eq_k_givenY = {}
        for i in range(X.shape[1]):
            K = X[:, i].max() + 1
            assert (X.min() == 0)
            tmpSum = np.ones((C, K)) * alpha
            for y in range(C):
                cython_inc_at(tmpSum[y, :], X[Y == y][:, i], 1)
                #         vals,counts=np.unique(X[Y==y][:i],return_counts=True)
                assert (tmpSum[y, :].sum() != 0)
                tmpSum[y, :] /= tmpSum[y, :].sum()
            log_pXi_eq_k_givenY[i] = np.log(tmpSum.T)

        self.C = C
        self.py = py

        self.log_pXi_eq_k_givenY = log_pXi_eq_k_givenY

    def predict_proba(self, X):

        logPY_given_x = np.zeros((X.shape[0], self.C))
        logPY_given_x += np.log(self.py)

        for y in range(C):
            for j in range(X.shape[1]):
                #                 np.add.at(logPY_given_x[y], )
                logPY_given_x[:, y] += self.log_pXi_eq_k_givenY[j][X[:, j], y]

        #         for i in range(X.shape[0]):
        #             for y in range(C):
        #                 logPY_given_x[y]+=np.log(self.py[y])
        #                 for j in range(X.shape[1]):
        #                     try:
        #                         logPY_given_x[y]+=self.log_pXi_eq_k_givenY[j][X[i,j], y]
        #                     except Exception as ex:
        #                         print(y, j, i)
        #                         raise(ex)

        py_given_x = np.exp(logPY_given_x)
        py_given_x /= py_given_x.sum(axis=1, keepdims=True)

        return py_given_x


class MeanClassifier():

    def __init__(self):
        pass

    def fit(self, X, Y, alpha=1):
        C = Y.max() + 1
        py = np.zeros(C)
        for y in range(C):
            py[y] = (Y == y).mean()

        self.C = C
        self.py = py

    def predict_proba(self, X):
        res = np.zeros((X.shape[0], self.C))
        res += self.py
        return res


class AllZeroClassifier():

    def __init__(self):
        pass

    def fit(self, X, Y, alpha=1):
        C = Y.max() + 1

        self.C = C

    def predict_proba(self, X):
        res = np.zeros((X.shape[0], self.C))
        res[:, 0] = 1
        return res

    def to_json(self):
        return {'params': None}
