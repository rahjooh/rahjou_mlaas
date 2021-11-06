import pydoc
import pydoc
import threading
from builtins import Exception
from collections import OrderedDict, defaultdict
from typing import List, Union, Optional, Any  # , Final
import scipy as sp

import numpy as np
import sklearn
import sklearn.preprocessing
from pydantic import BaseModel, validator, root_validator

from mlservice.common import util
from mlservice.common.util import compressArr


class Fitable(BaseModel):
    train_sw_colname: Optional[str]
    freezed: bool = False
    train_coldata_sample: dict = None

    def fit_(self, coldata_subset):
        pass

    @root_validator(pre=True)
    def fit_should_not_be_overriden(cls, values):
        if cls.fit != Fitable.fit:
            raise Exception(
                'The fit method can not be overriden to prevent error in handling of training set. Override fit_ method instead.')

        return values

    def fit(self,
            coldata):  # ** This method should not be overrided (unless making sure that train_sw_colname is handled properly)
        self.train_coldata_sample = {colname: coldata[colname][0:1] for colname in coldata}
        if self.train_sw_colname is None:
            self.fit_({colname: coldata[colname] for colname in coldata})
        else:
            self.fit_({colname: coldata[colname][coldata[self.train_sw_colname]] for colname in coldata})

    def fit_transform(self, coldata):
        self.fit(coldata)
        self.transform(coldata)

    def getRequiredColnames_fit(self):
        res = self.getRequiredColnames()
        if not (self.train_sw_colname is None):  # isinstance(self,Fitable) and
            print(self, self.train_sw_colname)
            res += [self.train_sw_colname]
        else:
            print(self)
        return res


class Stage(BaseModel):
    op: Optional[str]  # It can be specified in the group
    # input: Optional[Union[
    #     str, List[str], Dict[str, str]]]  # Single argument, sequence of arguments, named arguments (not used currently)
    input: Optional[Any]  # Single argument, sequence of arguments, named arguments (not used currently)
    output: Optional[Union[str, List[str]]]
    res: Optional[Any]
    warnings: Optional[Any]
    exception: Optional[Any]

    # runfit: bool = False
    # train_subset_sw: Optional[str]
    # exec_context: Optional[List[ExecContext]]

    def getRequiredColnames(self):
        res = []
        if self.input is None:
            return res
        if isinstance(self.input, list):
            res += self.input
        else:
            res += [self.input]
        #         res+=[self.output]
        return res

    def getChangedColnames(self):
        res = []
        if self.output is None:
            return res
        if isinstance(self.output, list):
            res += self.output
        else:
            res += [self.output]
        #         res+=[self.output]
        return res

    @validator('op')
    def operator_must_be_subclass_name(cls, v, values):
        if cls.__name__ != v:
            raise ValueError(f"'op' field does not match the class ({v}!={cls.__name__}).")
        return v
        # return self.__class__.__name__ == self.op

    def clean_to_def(self):
        self.res = None
        self.warnings = None
        self.exception = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not 'op' in data:
            self.op = self.__class__.__name__
        self.input = self.input if isinstance(self.input, list) else [self.input]
        # print(self.__class__.__name__)

    def getWarnings(self):
        if self.warnings == None:
            self.warnings = []
        return self.warnings


class Quantize(Stage, Fitable):
    splits: Optional[List[float]]
    len: Optional[int]
    method: Optional[str] = "decision_tree"
    label: Optional[str]
    model: Optional[Any]

    def getRequiredColnames(self):
        res = super().getRequiredColnames()
        # res += [self.label]
        return res

    def getRequiredColnames_fit(self):
        res = super().getRequiredColnames_fit()
        res += [self.label]
        return res

    def transform(self, coldata):
        # print('dbg', self.splits)
        if not self.splits is None:
            if isinstance(self.input, list):
                input = self.input[0]
            else:
                input = self.input

            coldata[self.output] = compressArr(np.searchsorted(self.splits, coldata[input]))
        else:
            x = np.hstack([coldata[colname][:, np.newaxis] for colname in self.input])
            coldata[self.output] = compressArr(self.model.apply(x))

    def fit_(self, coldata_subset):
        if self.freezed:
            return
        # print('dbg',coldata_subset[self.input], self.label, self.len)
        if self.method == 'decision_tree':
            if len(self.input) == 1:
                input = self.input[0]
            else:
                input = self.input

            if isinstance(input, str):
                self.splits = Quantize.quantize_using_decision_tree_1d(coldata_subset[input],
                                                                       coldata_subset[self.label], self.len)
                self.model = None
                self.len = len(self.splits)  # It seems that sometimes it change
            else:
                x = np.hstack([coldata_subset[colname][:, np.newaxis] for colname in input])
                self.model = Quantize.quantize_using_decision_tree(x, coldata_subset[self.label], self.len)
                self.splits = None
                # TODO: I don't know if there will be problems if 'len' is changed
        else:
            raise Exception(f'Unknown quantization method: {self.method}')

    def model_to_json(self):
        if self.splits is None:
            raise Exception('Not supported (you may have used more than one input field).')

        res = {
            'Quantize': {'input': self.input, 'output': self.output, 'splits': list(self.splits)}
        }
        return res

    def load_json(self, doc):
        self.input = doc['Quantize']['input']
        self.output = doc['Quantize']['output']
        self.splits = np.array([float(elem) for elem in doc['Quantize']['splits']])

    @staticmethod
    def quantize_using_decision_tree(x, y, num_splits):
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=num_splits)

        if (len(x.shape) == 1):
            tree.fit(x[:, np.newaxis], y)
        else:
            tree.fit(x, y)

        return tree

    @staticmethod
    def quantize_using_decision_tree_1d(x, y, num_splits):
        sortIndcs = np.argsort(x)  # np.argsort(coldata_x)
        x = x[sortIndcs]
        y = y[sortIndcs]

        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=num_splits)
        tree.fit(x[:, np.newaxis], y)

        allTresholds = np.sort(tree.tree_.threshold)
        allTresholds = allTresholds[allTresholds != -2]

        return allTresholds


class KBinsDiscretizerSparse(Stage, Fitable):
    """
    Sparse implementation of sklearn.preprocessing.KBinsDiscretizer.
    encode: 'ordinal', 'onehot' or 'onehot-dense'.
    columnvise: switch to use separate discretizer for each column.
    """
    discretizers: Optional[list]  # list[sklearn.preprocessing.KBinsDiscretizer]
    len: int
    columnvise: bool = True
    encode: str = 'onehot'
    strategy: str = 'quantile'

    def transform(self, coldata):
        if len(self.input) == 1:
            input = self.input[0]
        else:
            input = self.input

        arr = coldata[input]

        N = arr.shape[0]
        D = arr.shape[1]
        if D != len(self.discretizers):
            raise Exception(f'Sparse data diemnsion changed from training: {len(self.discretizers)} -> {D}')

        if not self.columnvise:
            rows, cols, vals = sp.sparse.find(arr)
            newvals = 1 + self.discretizers[0].transform(vals[:, np.newaxis])[:, 0]
        else:
            rows_conc = []
            cols_conc = []
            vals_conc = []

            for j in range(arr.shape[1]):
                col = arr[:, j]
                rows, cols, vals = sp.sparse.find(col)
                if vals.shape[0] != 0:
                    newvals = 1 + self.discretizers[j].transform(vals[:, np.newaxis])[:, 0]
                else:  # All zero case
                    newvals = vals

                cols_conc.append(np.ones(len(cols)) * j)
                rows_conc.append(rows)
                vals_conc.append(newvals)

            rows, cols, newvals = [np.concatenate(arr_list) for arr_list in [rows_conc, cols_conc, vals_conc]]

        newvals = util.compressArr(newvals)
        if self.encode == 'ordinal':
            newarr = sp.sparse.csc_matrix((newvals, (rows, cols)), shape=(N, D), dtype=newvals.dtype)
        else:  # 'onehot' or 'onehot-dense'
            newarr = sp.sparse.csc_matrix((np.ones(len(newvals)), (rows, cols * self.len + newvals)),
                                          shape=(N, D * self.len))
            if self.encode == 'onehot-dense':
                newarr = newarr.toarray()

        coldata[self.output] = compressArr(newarr)

    def fit_(self, coldata_subset):
        if self.freezed:
            return

        if len(self.input) == 1:
            input = self.input[0]
        else:
            input = self.input

        arr = coldata_subset[input]

        N = arr.shape[0]
        D = arr.shape[1]

        if not self.columnvise:
            self.discretizers = [None]
            rows, cols, vals = sp.sparse.find(arr)
            self.discretizers[0] = sklearn.preprocessing.KBinsDiscretizer(n_bins=self.len - 1, encode='ordinal',
                                                                          strategy=self.strategy)
            self.discretizers[0].fit(vals[:, np.newaxis])
        else:
            self.discretizers = [None] * D
            for j in range(arr.shape[1]):
                self.discretizers[j] = sklearn.preprocessing.KBinsDiscretizer(n_bins=self.len - 1, encode='ordinal',
                                                                              strategy=self.strategy)
                col = arr[:, j]
                rows, cols, vals = sp.sparse.find(col)
                self.discretizers[j].fit(vals[:, np.newaxis])

    def model_to_json(self):
        raise NotImplementedError()

    def load_json(self, doc):
        self.input = doc['Quantize']['input']
        self.output = doc['Quantize']['output']
        self.splits = np.array([float(elem) for elem in doc['Quantize']['splits']])

    @staticmethod
    def quantize_using_decision_tree(x, y, num_splits):
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=num_splits)

        if (len(x.shape) == 1):
            tree.fit(x[:, np.newaxis], y)
        else:
            tree.fit(x, y)

        return tree

    @staticmethod
    def quantize_using_decision_tree_1d(x, y, num_splits):
        sortIndcs = np.argsort(x)  # np.argsort(coldata_x)
        x = x[sortIndcs]
        y = y[sortIndcs]

        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=num_splits)
        tree.fit(x[:, np.newaxis], y)

        allTresholds = np.sort(tree.tree_.threshold)
        allTresholds = allTresholds[allTresholds != -2]

        return allTresholds


class Filter(Stage):
    conditions: dict  # Mongo like conditions (currently 'lt' and 'gt' supported)

    def transform(self, coldata):
        nColdata = next(iter(coldata.values())).shape[0]
        coldataSw = np.ones(nColdata, dtype=np.bool)
        varname = self.input
        varfilts = self.conditions
        for filttype, filtval in varfilts.items():
            if (filttype == 'lt'):
                coldataSw = np.logical_and(coldataSw, coldata[varname] < int(util.datetime2milis(filtval)))
            elif (filttype == 'gt'):
                coldataSw = np.logical_and(coldataSw, coldata[varname] > int(util.datetime2milis(filtval)))

        for colname in coldata:
            coldata[colname] = coldata[colname][coldataSw]


class Impute(Stage, Fitable):
    surrogate: Optional[float] = None
    method: str = 'mean'

    def transform(self, coldata):
        self.input = self.input if isinstance(self.input, list) else [self.input]
        tmp = coldata[self.input[0]].copy()
        tmp[np.isnan(tmp)] = self.surrogate
        coldata[self.output] = tmp

    def fit_(self, coldata_subset):
        if self.freezed:
            return
        if self.method == 'mean':
            self.input = self.input if isinstance(self.input, list) else [self.input]
            self.surrogate = np.nanmean(coldata_subset[self.input[0]])
        else:
            raise Exception(f'Unknown imputation method: {self.method}')

    def model_to_json(self):
        res = {
            'Impute': {'input': self.input, 'output': self.output, 'surrogate': self.surrogate}
        }
        return res

    def load_json(self, doc):
        self.input = doc['Impute']['input']
        self.output = doc['Impute']['output']
        self.surrogate = float(doc['Impute']['surrogate'])


class Onehot(Stage, Fitable):
    lengths: Optional[List[int]]

    def validate_data(self, coldata):
        try:
            for colname in self.input:
                assert (not coldata[colname].dtype == np.float64)
                assert (not coldata[colname].dtype == np.float32)
                assert (not coldata[colname].dtype == np.float)
                assert (not coldata[colname].min() < 0)
        except Exception as ex:
            raise Exception(f'Error validating data for {self}') from ex

    def fit_(self, coldata_subset):
        self.validate_data(coldata_subset)
        if self.freezed:
            return
        # print('dbg',next(iter(coldata_subset.values())).shape[0])
        self.lengths = [coldata_subset[colname].max() + 1 for colname in
                        self.input]  # len(np.unique(coldata_subset[self.input]))

    def transform(self, coldata):
        self.input = self.input if isinstance(self.input, list) else [self.input]
        self.validate_data(coldata)
        from scipy.sparse import csr_matrix
        n = coldata[self.input[0]].shape[0]
        #         M = self.len  # max(self.len, coldata[self.input].max() + 1)
        if self.output in coldata:
            self.getWarnings().append(f'Column rewrite: {self.output}')

        data = np.ones(n)
        row_ind = np.arange(n)

        multiplier = 1
        col_ind = np.zeros(n, dtype=np.uint)
        valid_col_sw = np.ones(n, dtype=np.bool)
        for i in list(range(len(self.input)))[::-1]:
            colname = self.input[i]
            M = self.lengths[i]
            arr = coldata[colname].astype(np.uint)  # += does not wotk if it is np.int
            col_ind += arr * multiplier
            valid_col_sw &= arr < M
            multiplier *= M

        totalM = multiplier

        if (~valid_col_sw).any():
            print(
                f'Warning - Onehot({self.input}): {(~valid_col_sw).mean()} of data is lost in Onehot as not seen in training time')
        res = csr_matrix((data[valid_col_sw], (row_ind[valid_col_sw], col_ind[valid_col_sw])), shape=(n, totalM),
                         dtype=np.uint8)

        coldata[self.output] = res

    ######### OLD: non sparse

    #         # print('dbg', next(iter(coldata.values())).shape[0])
    #         n = coldata[self.input].shape[0]
    #         M = max(self.len, coldata[self.input].max() + 1)
    #         if self.output in coldata:
    #             self.getWarnings().append(f'Column rewrite: {self.output}')

    #         res = np.zeros((n, M), np.uint8)
    #         res[np.arange(n), coldata[self.input]] = 1
    #         coldata[self.output]=res

    def model_to_json(self):
        res = {
            'Onehot': {'input': self.input, 'output': self.output, 'dimsLen': self.lengths}
        }
        return res

    def load_json(self, doc):
        self.input = doc['Onehot']['input']
        self.output = doc['Onehot']['output']
        self.lengths = int(doc['Onehot']['dimsLen'])


class Sort(Stage):
    reversed: bool = False

    @validator('output')
    def output_must_be_empty(cls, v, values):
        if not v is None and len(v) > 0:
            raise ValueError('Sort does not support "output" argument.')

    # def __init__(self, **data: Any):
    #     super().__init__(**data)
    #     if len(self.output) > 0:
    #         raise Exception('Sort does not support "output" argument.')

    def transform(self, coldata):
        sortIndcs = np.lexsort([coldata[colname] for colname in self.input])
        if self.reversed:
            sortIndcs = sortIndcs[::-1]
        for colname in coldata:
            coldata[colname] = coldata[colname][sortIndcs]


class Split(Stage):
    ratios: List[float]
    random_permute: Optional[bool] = False
    trim_others = True  # remove the rows not present in any split

    @validator('input')
    def input_must_be_empty(cls, v, values):
        if not v is None and len(v) > 0:
            raise ValueError('Split does not support "input" argument.')

    def transform(self, coldata):
        nTotal = (next(iter(coldata.values()))).shape[0]

        assert sum(self.ratios) <= 1

        all_n = [int(ratio * nTotal) for ratio in self.ratios]
        all_sw = [np.zeros(nTotal, dtype=np.bool) for ratio in self.ratios]
        indcs = np.arange(nTotal)

        if (self.random_permute):
            randIndcs = np.random.permutation(np.arange(nTotal))
            indcs = indcs[randIndcs]

        prev_ind = 0
        for i in np.arange(0, len(all_n))[::-1]:
            # for cur_n in all_n[:-1]:
            cur_n = all_n[i]
            if (prev_ind == 0):
                cur_indcs = indcs[prev_ind - cur_n:]
            else:
                cur_indcs = indcs[prev_ind - cur_n:prev_ind]
            prev_ind = prev_ind - cur_n
            all_sw[i][cur_indcs] = True

        coldata_sw = np.zeros_like(all_sw[0])
        for i in range(len(all_sw)):
            if self.output[i] in coldata:
                self.getWarnings().append(f'Column rewrite: {self.output[i]}')
            coldata[self.output[i]] = all_sw[i]
            coldata_sw = np.logical_or(coldata_sw, all_sw[i])

        if self.trim_others:
            for colname in coldata:
                coldata[colname] = coldata[colname][coldata_sw]


class Compressor(Stage, Fitable):
    old2new: Any
    new2old: Any
    isIdentity: bool = False

    def transform(self, coldata):

        if self.old2new is None or self.isIdentity:
            coldata[self.output] = coldata[
                self.input]  # If we don't suppose immutability this line is unsafe and might require 'copy'
        else:
            coldata[self.output] = self.old2new[compressArr(coldata[self.input])]

    def fit_(self, coldata_subset):
        if self.freezed:
            return
        # print('dbg',coldata_subset[self.input], self.label, self.len)
        x = compressArr(coldata_subset[self.input])

        if not x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
            old2new = None
            new2old = None
            return

        self.new2old = np.unique(x)
        self.old2new = np.ones(self.new2old.max() + 1, dtype=self.new2old.dtype) * -1
        self.old2new[self.new2old] = np.arange(self.new2old.shape[0])
        self.old2new = compressArr(self.old2new)

        if (self.old2new == np.arange(self.old2new.shape[0])).all():
            self.isIdentity = True
        else:
            self.isIdentity = False

    def model_to_json(self):
        if self.isIdentity:
            return []

        res = {
            'Map': {'input': self.input, 'output': self.output, 'old2new': list(self.old2new)}
        }
        return res

    def load_json(self, doc):
        raise NotImplementedError()


class Pipeline(Stage, Fitable):
    stages: List[Stage]

    #     def __init__(self,stages):
    #         self.stages=stages

    def getRequiredColnames(self):
        res = []
        prevs = set()
        for stage in self.stages:
            inps = stage.getRequiredColnames()
            for colname in inps:
                if not colname in prevs:
                    res.append(colname)
                    if colname is None:
                        print(stage)
                        raise Exception()
            # if isinstance(stage.output, list):
            #     raise Exception('Not implemented')

            prevs |= set(stage.getChangedColnames())
        #             print(prevs)

        return list(OrderedDict.fromkeys(res))  # sorted(list(set(res)))

    def getRequiredColnames_fit(self):
        res = []
        prevs = set()
        for stage in self.stages:
            inps = stage.getRequiredColnames_fit()
            for colname in inps:
                if not colname in prevs:
                    res.append(colname)
                    if colname is None:
                        print(stage)
            if isinstance(stage.output, list):
                raise Exception('Not implemented')

            prevs |= set(stage.getChangedColnames())
        #             print(prevs)

        return list(OrderedDict.fromkeys(res))  # sorted(list(set(res)))

    def getChangedColnames(self):
        res = []
        for stage in self.stages:
            res += stage.getChangedColnames()

        return res

    def assertAllColnamesArePresent(self, coldata):
        for colname in self.getRequiredColnames():
            if not colname in coldata:
                raise Exception(f'Column missing: {colname}')

    def fit_transform(self, coldata,
                      verbose=False):  # It should override this method because fit and transform of stages should be in order
        self.assertAllColnamesArePresent(coldata)
        self.train_coldata_sample = {colname: coldata[colname][0:1] for colname in coldata}
        for stage in self.stages:
            try:
                if verbose:
                    print('eval stage:', stage)
                if isinstance(stage, Fitable):
                    stage.fit_transform(coldata)
                else:
                    stage.transform(coldata)
            except Exception as e:
                raise Exception(f'stage failed: {stage}') from e

    def fit_(self, coldata_subset):
        raise Exception(
            'For pipelines the fit method can not be evaluated separately as the stages may depend on each other. Use fit_transform instead.')
        # TODO: I can make a temporal copy of the data and then call fit_transform
        self.assertAllColnamesArePresent(coldata)
        for stage in self.stages:
            if isinstance(stage, Fitable):
                stage.fit(coldata_subset)

    def transform(self, coldata):
        self.assertAllColnamesArePresent(coldata)
        for stage in self.stages:
            stage.transform(coldata)

    def model_to_json(self):
        res = []
        for stage in self.stages:
            if 'model_to_json' in dir(stage):
                childJson = stage.model_to_json()
                if isinstance(childJson, list):
                    for elem in childJson:
                        res.append(elem)
                else:
                    res.append(childJson)
        return res

    def load_json(self, doc):
        self.stages = []
        for subdoc in doc:
            stageName = next(iter(subdoc.keys()))
            stage = eval(stageName + '.construct()')  # pydantic way to construct without validation
            stage.load_json(subdoc)
            self.stages.append(stage)

    def iter_stages(self):
        for stage in self.stages:
            if isinstance(stage, Pipeline):
                for ch_stage in stage.iter_stages():
                    assert (not isinstance(ch_stage, Pipeline))  # Means bug
                    yield ch_stage
            else:
                yield stage


lock = threading.Lock()


def make_XY(coldata, featuresColnames, labelColname=None):
    import scipy.sparse
    if isinstance(featuresColnames, str):
        featuresColnames = [featuresColnames]

    if any([isinstance(coldata[colname], scipy.sparse.spmatrix) for colname in featuresColnames]):
        data_fixed_dim = [coldata[colname] for colname in featuresColnames]
        for i in range(len(data_fixed_dim)):
            if len(data_fixed_dim[i].shape) == 1:
                data_fixed_dim[i] = data_fixed_dim[i][:, np.newaxis]
        X = compressArr(scipy.sparse.hstack(tuple(data_fixed_dim)))
    else:
        X = compressArr(np.column_stack(tuple(coldata[colname] for colname in featuresColnames)))
    if not labelColname is None:
        Y = coldata[labelColname].astype(np.uint8)
        return X, Y
    else:
        return X


class Learner(Stage, Fitable):
    model: str
    params: Optional[dict] = {}
    label: str
    clf: Optional[Any]

    def getRequiredColnames(self):
        res = super().getRequiredColnames()
        # res += [self.label]
        return res

    def getRequiredColnames_fit(self):
        res = super().getRequiredColnames()
        res += [self.label]
        return res

    def fit_(self, coldata_subset):
        learner = self
        with lock:
            #             print(learner.model)
            clfClass = pydoc.locate(learner.model, forceload=0)

        Xtrain, Ytrain = make_XY(coldata_subset, learner.input, learner.label)

        data_stats = {'n_train': len(Ytrain)}

        data_stats['class_ratios'] = {'train': Ytrain.mean(axis=0).tolist()}

        if len(np.unique(Ytrain)) > 2:
            raise Exception(
                'Currently only binray classification is supported.')  # The only problem is the json converor and which now behave in 1 dimension

        clf = clfClass(**learner.params)
        clf.fit(Xtrain, Ytrain)

        self.clf = clf

    def transform(self, coldata):
        X = make_XY(coldata, self.input, None)
        #         print(np.dot(self.clf.coef_[0],X)[0])
        #         print(np.dot(self.clf.coef_[0],X)[0]+self.clf.intercept[0])
        coldata[self.output] = self.clf.predict_proba(X)[:, 1]

    to_json_converters = {
        'sklearn.linear_model.logistic.LogisticRegression': lambda clf:
        {
            'model': 'sklearn.linear_model.logistic.LogisticRegression',
            'intercept': clf.intercept_[0],
            'coef': clf.coef_[0]
        },
        'sklearn.linear_model.LogisticRegression': lambda clf:
        {
            'model': 'sklearn.linear_model.logistic.LogisticRegression',
            'intercept': clf.intercept_[0],
            'coef': clf.coef_[0]
        }

    }

    def model_to_json(self):
        cls_name = self.model  # util.class_fullname(self.clf)

        if ('to_json' in dir(self.clf)):
            json = self.clf.to_json()
        elif cls_name in self.to_json_converters:
            json = self.to_json_converters[cls_name](self.clf)
        else:
            json = {'error': f'{type(self.clf)} does not implement a to_json(self) method'}

        json.update({'input': self.input, 'output': self.output})
        return {'Learner': json}

    def load_json(self, doc):
        self.input = doc['Learner']['input']
        self.output = doc['Learner']['output']
        self.model = doc['Learner']['model']
        if self.model in ['sklearn.linear_model.logistic.LogisticRegression',
                          'sklearn.linear_model.LogisticRegression']:
            with lock:
                clfClass = pydoc.locate(self.model, forceload=0)

            self.clf = clfClass()
            self.clf.intercept_ = np.array([doc['Learner']['intercept']])
            self.clf.coef_ = np.array([doc['Learner']['coef']])
        else:
            raise Exception(f'Unknown learner: {self.model}')


class Spline(Stage):
    x: List[float]  # Optional[List[float]]
    y: List[float]  # Optional[List[float]]

    @validator('x')
    def x_must_be_sorted(cls, v, values):
        if not util.is_lexsorted([np.array(v)]):
            raise ValueError('x_must_be_sorted.')

        return v

    #     def __init__(self,inp,output,x,y):
    #         sortIndcs=np.argsort(x)
    #         x=x[sortIndcs]
    #         y=y[sortIndcs]

    #         self.input=inp
    #         self.output=output
    #         self.x=x
    #         self.y=y

    def transform(self, coldata):
        coldata[self.output] = self.transform_xnew_(coldata[self.input])

    def transform_xnew_(self, xnew):
        x = np.array(self.x)
        y = np.array(self.y)
        insertInd = np.searchsorted(x, xnew)
        prevInd = insertInd - 1
        nextInd = insertInd
        prevInd[prevInd == -1] += 1
        nextInd[nextInd == len(self.y)] -= 1

        prevx = x[prevInd]
        nextx = x[nextInd]
        prevy = y[prevInd]
        nexty = y[nextInd]

        alpha = np.zeros(xnew.shape)
        alpha[nextx != prevx] = (nextx - xnew)[nextx != prevx] / (nextx - prevx)[nextx != prevx]

        return alpha * prevy + (1 - alpha) * nexty

    def model_to_json(self):
        if self.x is None:
            raise Exception('Not supported.')

        res = {
            'Spline': {'input': self.input, 'output': self.output, 'x': list(self.x), 'y': list(self.y)}
        }
        return res

    def load_json(self, doc):
        self.input = doc['Spline']['input']
        self.output = doc['Spline']['output']
        self.x = np.array([float(elem) for elem in doc['Spline']['x']])
        self.y = np.array([float(elem) for elem in doc['Spline']['y']])


class ExtractColumn(Stage):
    """
    This stage is used to convert a column which contains 2d data to a simple 1d column.
    It is related to "VectorSlicer", "VectorDissassmbler" and "getItem" functionalities in Spark.
    """

    index: int

    def transform(self, coldata):
        self.input = self.input if isinstance(self.input, list) else [self.input]
        inp = self.input[0]
        coldata[self.output] = np.asarray(coldata[inp][:, self.index].todense())[:, 0]
