"""
The schemas for all learning steps in mongo + their implementation
"""

import inspect
import pydoc
import types
from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, validator, Field
from pydantic.dataclasses import dataclass
from typing import List, Union, Optional, Any, TypeVar, Dict  # , Final
from abc import ABC, abstractmethod

import copy


def make_hash(o):
    """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  """

    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):
        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))


class DatasetDef(BaseModel):
    type: str
    read_func: Optional[str]

    server: Optional[str]
    database: Optional[str]
    collection: Optional[str]
    max_read_size: int = -1
    query: Optional[
        Dict]  # Final[Optional[Dict]] # Good to be defined final as is used inside hash, but commented out since Final not avilable in python 3.7

    alias: Optional[str]
    merge_key: Optional[str]
    cache_valid_seconds: int

    extra_info: Optional[dict]

    def _important_fields(self):
        return (self.type, self.read_func, self.server, self.database, self.collection, self.alias, self.extra_info,
                self.max_read_size, self.query)

    def __hash__(self) -> int:
        # All the fields which affect the coldata generated
        return make_hash(self._important_fields()).__hash__()

    def __eq__(self, other):
        return make_hash(self._important_fields()) == make_hash(other._important_fields())
        # return self._important_fields() == other._important_fields()
    # def __hash__(self):
    #     return (self.type).__hash__()


AnyArgType: TypeVar = Union[float, int, bool, str]


class ExecContext(BaseModel):
    func: str
    subset_sw: Optional[str]  # Note: "None" value means the whole set
    res: Optional[Any]


class Fitable(BaseModel):
    train_sw_colname: Optional[str]
    freezed: bool = False

    def fit(self, coldata_subset):
        pass


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
        # print(self.__class__.__name__)

    def getWarnings(self):
        if self.warnings == None:
            self.warnings = []
        return self.warnings


import numpy as np
from mlservice.common import util


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
    surrogate: float
    method: str = 'mean'

    def transform(self, coldata):
        tmp = coldata[self.input].copy()
        tmp[np.isnan(tmp)] = self.surrogate
        coldata[self.output] = tmp

    def fit(self, coldata_subset):
        if self.freezed:
            return
        if self.method == 'mean':
            self.surrogate = np.nanmean(coldata_subset[self.input])
        else:
            raise Exception(f'Unknown imputation method: {self.method}')


class Quantize(Stage, Fitable):
    splits: Optional[List[float]]
    len: Optional[int]
    method: Optional[str] = "decision_tree"
    label: Optional[str]

    def transform(self, coldata):
        # print('dbg', self.splits)
        coldata[self.output] = np.searchsorted(self.splits, coldata[self.input])

    def fit(self, coldata_subset):
        if self.freezed:
            return
        # print('dbg',coldata_subset[self.input], self.label, self.len)
        if self.method == 'decision_tree':
            self.splits = Quantize.quantize_using_decision_tree(coldata_subset[self.input], coldata_subset[self.label],
                                                                self.len)
            self.len = len(self.splits)  # It seems that sometimes it change
        else:
            raise Exception(f'Unknown imputation method: {self.method}')

    @staticmethod
    def quantize_using_decision_tree(x, y, num_splits):
        sortIndcs = np.argsort(x)  # np.argsort(coldata_x)
        x = x[sortIndcs]
        y = y[sortIndcs]

        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier("entropy", max_leaf_nodes=num_splits)
        tree.fit(x[:, np.newaxis], y)

        allTresholds = np.sort(tree.tree_.threshold)
        allTresholds = allTresholds[allTresholds != -2]

        return allTresholds


class Onehot(Stage, Fitable):
    len: int

    def fit(self, coldata_subset):
        if self.freezed:
            return
        # print('dbg',next(iter(coldata_subset.values())).shape[0])
        self.len = len(np.unique(coldata_subset[self.input]))

    def transform(self, coldata):
        # print('dbg', next(iter(coldata.values())).shape[0])
        n = coldata[self.input].shape[0]
        M = max(self.len, coldata[self.input].max() + 1)
        if self.output in coldata:
            self.getWarnings().append(f'Column rewrite: {self.output}')
        coldata[self.output] = np.zeros((n, M), np.uint8)
        coldata[self.output][np.arange(n), coldata[self.input]] = 1


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
    def output_must_be_empty(cls, v, values):
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


class CustomStage(
    Stage):  # Most general custom stage so far: applying a function which gets whole coldata and some parameters and change the coldata as it wants
    """
    Note: the argument named 'extra_res' is a dict which provides the possibility to return extra run info.
    """
    func: str
    extra_res: Optional[Any]

    def execute(self, coldata):

        aug_coldata = {}

        funcc = pydoc.locate(self.func)
        extra_res = {}

        if (isinstance(self.input, dict)):
            args = self.input

            if isinstance(funcc, types.FunctionType) and 'extra_res' in inspect.getfullargspec(funcc).args:
                # Note the first condition is needed as the second condition throws exception for cpython functions (like many numpy functions)
                args['extra_res'] = extra_res

            res = funcc(coldata, **args)  # , extra_res=extra_res)

        if isinstance(self.input, list):
            args = self.input

            if isinstance(funcc, types.FunctionType) and 'extra_res' in inspect.getfullargspec(funcc).args:
                # Note the first condition is needed as the second condition throws exception for cpython functions (like many numpy functions)
                res = funcc(coldata, *args, extra_res=extra_res)
            else:
                res = funcc(coldata, *args)

        if extra_res != {}:
            self.extra_res = extra_res

        if not isinstance(res, tuple):  # For single outputs
            res = tuple([res])
        assert len(res) == len(self.output), \
            f'Wrong augementation: incorrect number of outputs ({len(res)} != {len(self.output)})'

        for i, colname in enumerate(self.output):
            # assert res[i].shape[0] == n, \
            #     'Wrong augementation: Number of rows should be the same for all columns of coldata'
            aug_coldata[colname] = res[i]

        for colname in aug_coldata:
            if colname in coldata:
                self.getWarnings().append(f'Column rewrite: {colname}')

            assert not colname in coldata, \
                f'{colname} already exists in coldata. Rewriting on columns is a bad idea (immutability might be needed in the future)'
            coldata[colname] = util.compressArr(aug_coldata[colname])

        return extra_res

    def clean_to_def(self):
        super().clean_to_def()
        self.extra_res = None


class CustomStage_deprecated(Stage):  # The incomplete implementaion which allowed subset_sw
    """
    Note: the argument named 'extra_res' is a dict which provides the possibility to return extra run info.
    """
    exec_context: List[ExecContext]

    def execute(self, coldata):

        aug_coldata = {}
        n = next(iter(coldata.values())).shape[0]
        for colname in self.output:
            aug_coldata[colname] = np.zeros(n)
            aug_coldata[colname].fill(np.nan)

        for i in range(len(self.exec_context)):
            func = self.exec_context[i].func
            cond = self.exec_context[i].subset_sw
            if cond is None:
                coldata_sw = np.ones(n, dtype=np.bool)
            else:
                coldata_sw = coldata[cond]
            # for func, cond in self.exec_plan:

            funcc = pydoc.locate(func)
            extra_res = {}

            if (isinstance(self.input, dict)):
                args = self.input

                if isinstance(funcc, types.FunctionType) and 'extra_res' in inspect.getfullargspec(funcc).args:
                    args['extra_res'] = extra_res

                res = funcc(coldata, **args)  # , extra_res=extra_res)

            if isinstance(self.input, list):
                args = self.input

                if isinstance(funcc, types.FunctionType) and 'extra_res' in inspect.getfullargspec(funcc).args:
                    # Note the first condition is needed as the second condition throws exception for cpython functions (like many numpy functions)
                    res = funcc(coldata, *args, extra_res=extra_res)
                else:
                    res = funcc(coldata, *args)

            if extra_res != {}:
                self.exec_context[i].res = extra_res

            if not isinstance(res, tuple):  # For single outputs
                res = tuple([res])
            assert len(res) == len(self.output), \
                f'Wrong augementation: incorrect number of outputs ({len(res)} != {len(self.output)})'

            for i, colname in enumerate(self.output):
                # assert res[i].shape[0] == n, \
                #     'Wrong augementation: Number of rows should be the same for all columns of coldata'
                aug_coldata[colname][coldata_sw] = res[i]

        for colname in aug_coldata:
            assert not colname in coldata, \
                f'{colname} already exists in coldata. Rewriting on columns is a bad idea (immutability might be needed in the future)'
            coldata[colname] = util.compressArr(aug_coldata[colname])

        return self.exec_context

    def clean_to_def(self):
        super().clean_to_def()
        for i in range(len(self.exec_context)):
            self.exec_context[i].res = None


class CustomAugment(Stage):  # Calling methods for augmenting the coldata by adding some new columns
    """
    Note: the argument named 'extra_res' is a dict which provides the possibility to return extra run info.
    """
    exec_context: List[ExecContext]

    def execute(self, coldata):

        aug_coldata = {}
        n = next(iter(coldata.values())).shape[0]
        for colname in self.output:
            aug_coldata[colname] = np.zeros(n)
            aug_coldata[colname].fill(np.nan)

        for i in range(len(self.exec_context)):
            func = self.exec_context[i].func
            cond = self.exec_context[i].subset_sw
            if cond is None:
                coldata_sw = np.ones(n, dtype=np.bool)
            else:
                coldata_sw = coldata[cond]
            # for func, cond in self.exec_plan:

            funcc = pydoc.locate(func)
            extra_res = {}

            if (isinstance(self.input, dict)):
                args = {}
                for argname, val in self.input.items():
                    if isinstance(val, str):
                        args[argname] = coldata[val][coldata_sw]
                    else:
                        args[argname] = val

                if 'extra_res' in inspect.getfullargspec(funcc).args:
                    args['extra_res'] = extra_res

                res = funcc(**args)  # , extra_res=extra_res)

            if isinstance(self.input, list):
                args = []
                for val in self.input:
                    if isinstance(val, str):
                        args.append(coldata[val][coldata_sw])
                    else:
                        args.append(val)

                if isinstance(funcc, types.FunctionType) and 'extra_res' in inspect.getfullargspec(funcc).args:
                    # Note the first condition is needed as the second condition throws exception for cpython functions (like many numpy functions)
                    res = funcc(*args, extra_res=extra_res)
                else:
                    res = funcc(*args)

            if extra_res != {}:
                self.exec_context[i].res = extra_res

            if not isinstance(res, tuple):  # For single outputs
                res = tuple([res])
            assert len(res) == len(self.output), \
                f'Wrong augementation: incorrect number of outputs ({len(res)} != {len(self.output)})'
            for i, colname in enumerate(self.output):
                # assert res[i].shape[0] == n, \
                #     'Wrong augementation: Number of rows should be the same for all columns of coldata'
                aug_coldata[colname][coldata_sw] = res[i]

        for colname in aug_coldata:
            if colname in coldata:
                self.getWarnings().append(f'Column rewrite: {colname}')
            assert not colname in coldata, \
                f'{colname} already exists in coldata. Rewriting on columns is a bad idea (immutability might be needed in the future)'
            coldata[colname] = util.compressArr(aug_coldata[colname])

        return self.exec_context

    def clean_to_def(self):
        super().clean_to_def()
        for i in range(len(self.exec_context)):
            self.exec_context[i].res = None


class CustomFunction(Stage):
    exec_context: List[ExecContext]

    def execute(self, coldata):
        for i in range(len(self.exec_context)):
            func = self.exec_context[i].func
            cond = self.exec_context[i].subset_sw
            # for func, cond in self.exec_plan:
            args = {}
            for argname, val in self.input.items():
                if isinstance(val, str):
                    args[argname] = coldata[val][coldata[cond]]
                else:
                    args[argname] = val

            self.exec_context[i].res = pydoc.locate(func)(**args)

        return self.exec_context

    def clean_to_def(self):
        super().clean_to_def()
        for i in range(len(self.exec_context)):
            self.exec_context[i].res = None


class Learner(Stage):
    model: str
    params: Optional[dict] = {}
    label: str

    # def transform(self, coldata):
    #     pass


AnyStageType: TypeVar = Union[
    Filter, Impute, Quantize, Onehot, Learner, Sort, Split, CustomFunction, CustomAugment, CustomStage]


class StageGroup(BaseModel):
    op: Optional[str]
    repeat: List[AnyStageType]

    def __init__(self, **data: Any):
        if (not 'repeat' in data):
            raise ValueError('"repeat" field is missing')
        for key, val in data.items():
            if (key == 'repeat'):
                continue
            for elem in data['repeat']:
                if (not key in elem):
                    elem[key] = val

        super().__init__(**data)

    def fit(self, coldata_subset):
        raise Exception('Not implemented due to some problems...')
        for stage in self.repeat:
            # if isinstance(stage,Fitable):
            #     cond = stage.train_sw_colname
            #     coldata_sw = None
            #     if not cond is None:
            #         coldata_sw = coldata_subset[cond]
            #
            #     stage.fit(coldata_get_subset(coldata, coldata_sw, stage.input))
            stage.fit(coldata_subset)

    def transform(self, coldata):
        for stage in self.repeat:
            stage.transform(coldata)


class JobDef(BaseModel):
    datasets: List[DatasetDef]
    # max_data_size: int
    pre_sort_colname: Optional[str]
    train_ratio: float
    test_ratio: Optional[float]
    pipeline: List[Union[StageGroup, AnyStageType]]
    # evaluation: List[str]


class LearningJob(BaseModel):
    id: Optional[Any] = Field(..., alias='_id')  # _id:Optional[ObjectId]
    schema_revision: int
    job_name: str
    job_def: JobDef
    status: Optional[str]
    last_modified: Optional[datetime]
    data_stats: Optional[Dict]
    res: Optional[Dict[str, Any]]

    def clean_to_def(self):
        self.status = None
        self.last_modified = None
        self.data_stats = None
        self.res = None

        for elem in self.job_def.pipeline:
            elem.clean_to_def()


def test():
    raise Exception()
    job_def = JobDef(datasets=datasets, pipeline=pipeline, train_ratio=0.9, pre_sort_colname=None, )
    LearningJob(schema_revision=1, job_name='test', job_def=job_def)


def test_sum():
    assert sum([1, 2, 3]) == 3, "Should be 6"
