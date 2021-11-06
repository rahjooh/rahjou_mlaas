import pydoc
import pydoc
import threading
from collections import OrderedDict
from typing import List, Union, Optional, Any  # , Final

import numpy as np
from pydantic import BaseModel, validator, root_validator
from pyspark.sql.types import DoubleType

from mlservice.common import util
from mlservice.common.util import compressArr

import pandas as pd

from pyspark import SparkContext
# $example on$
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
import pyspark.sql.dataframe


def get_spark():
    spark = SparkSession.builder \
        .master('local') \
        .appName('mls') \
        .config('spark.executor.memory', '5gb') \
        .config("spark.cores.max", "6") \
        .getOrCreate()

    return spark


from pyspark.sql.functions import monotonically_increasing_id


def spark_removeDotFromColnames(colnames):
    return [colname.replace(".", "#") for colname in colnames]


def spark_encode_colnames(colnames):
    assert not any(['#' in colname for colname in colnames])  # Guarantee decodeability
    return [colname.replace(".", "#") for colname in colnames]


def spark_decode_colnames(colnames):
    assert not any(['.' in colname for colname in colnames])  # violates safe encode/decode
    return [colname.replace("#", ".") for colname in colnames]


class SparkColdata():
    spark_id_column = None

    def __init__(self, spark_df):
        spark_df = spark_df.withColumn("_spark_id", monotonically_increasing_id())
        if any(['.' in colname for colname in spark_df.columns]):
            spark_df = spark_df.toDF(*spark_encode_colnames(spark_df.columns))

        self.setdf(spark_df)

    def setdf(self, spark_df):
        assert ('_spark_id' in spark_df.columns)
        spark_df = spark_df.sort('_spark_id')
        self.df = spark_df
        # self.spark_id_column = None

    def get_spark_id_column(self):
        if self.spark_id_column is None:
            self.spark_id_column = util.coldata_from_pandas(self.df.select('_spark_id').toPandas())['_spark_id']
        return self.spark_id_column

    def __setitem__(self, key, value):
        print(f'*WARNING: conversion between numpy and spark (set {key}).')
        key = spark_encode_colnames([key])[0]
        otherdf = get_spark().createDataFrame(pd.DataFrame({"_spark_id": self.get_spark_id_column(), key: value}))
        res = self.df
        if key in res.columns:
            res = res.drop(key)
        res = res.join(otherdf, '_spark_id', 'inner')
        res = res.sort('_spark_id')
        self.df = res

    #         pass #df3 = df2.join(df1, "id", "outer").drop("id")

    def __getitem__(self, key):
        print(f'*WARNING: conversion between numpy and spark (get {key}).')
        spark_key = spark_encode_colnames([key])[0]
        if not isinstance(key, str):
            raise Exception(f'key should be string ({key})')
        assert (isinstance(key, str))
        keys = list(set([spark_key] + ['_spark_id']))
        tmp = util.coldata_from_pandas(self.df.select(keys).toPandas())

        assert (np.all(tmp['_spark_id'] == self.get_spark_id_column()))
        return tmp[spark_key]

    def __iter__(self):
        ''' Returns the Iterator object '''
        return spark_decode_colnames(self.df.columns).__iter__()


class Fitable(BaseModel):
    train_sw_colname: Optional[str]
    freezed: bool = False

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

        assert (isinstance(coldata, SparkColdata))

        if self.train_sw_colname is None:
            self.fit_(coldata)
        else:
            self.fit_(SparkColdata(coldata.df[coldata.df[spark_encode_colnames([self.train_sw_colname])[0]]]))

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

    @validator('input')
    def tmp(cls, v, values):
        if isinstance(v, list) and len(v) > 1:
            raise ValueError(f"Spark quantizer just accepts a single input field.")
        return v
        # return self.__class__.__name__ == self.op

    def getRequiredColnames(self):
        res = super().getRequiredColnames()
        res += [self.label]
        return res

    def transform(self, coldata):
        assert (isinstance(coldata, SparkColdata))

        util.tic()
        print(f'(transform) converting data to array: {self.getRequiredColnames()}')
        coldata_arrs = {colname: coldata[colname] for colname in self.getRequiredColnames()}
        # util.coldata_from_pandas(coldata.select(*self.getRequiredColnames_fit()).toPandas())
        print(f'(transform) Finished converting data to array: {self.getRequiredColnames_fit()}')
        util.toc()

        # print('dbg', self.splits)
        if not self.splits is None:
            if isinstance(self.input, list):
                input = self.input[0]
            else:
                input = self.input

            coldata[self.output] = compressArr(np.searchsorted(self.splits, coldata_arrs[input]))
        else:
            x = np.hstack([coldata_arrs[colname][:, np.newaxis] for colname in self.input])
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

            util.tic()
            print(f'Converting data to array: {self.getRequiredColnames_fit()}')
            coldata_subset_arrs = {colname: coldata_subset[colname] for colname in self.getRequiredColnames_fit()}
            print(f'Finished converting data to array: {self.getRequiredColnames_fit()}')
            util.toc()

            if isinstance(input, str):
                self.splits = Quantize.quantize_using_decision_tree_1d(coldata_subset_arrs[input],
                                                                       coldata_subset_arrs[self.label], self.len)
                self.model = None
                self.len = len(self.splits)  # It seems that sometimes it change
            else:
                x = np.hstack([coldata_subset_arrs[colname][:, np.newaxis] for colname in input])
                self.model = Quantize.quantize_using_decision_tree(x, coldata_subset_arrs[self.label], self.len)
                self.splits = None
                # TODO: I don't know if there will be problems if 'len' is changed
        else:
            raise Exception(f'Unknown imputation method: {self.method}')

    def model_to_json(self):
        if self.splits is None:
            raise Exception('Not supported (you may have used more than one input field).')

        res = {
            'Quantize': {'input': self.input, 'output': self.output, 'splits': list(self.splits)}
        }
        return res

    @staticmethod
    def quantize_using_decision_tree(x, y, num_splits):
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier("entropy", max_leaf_nodes=num_splits)

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
        tree = DecisionTreeClassifier("entropy", max_leaf_nodes=num_splits)
        tree.fit(x[:, np.newaxis], y)

        allTresholds = np.sort(tree.tree_.threshold)
        allTresholds = allTresholds[allTresholds != -2]

        return allTresholds


class Filter(Stage):
    conditions: dict  # Mongo like conditions (currently 'lt' and 'gt' supported)

    def transform(self, coldata):
        raise NotImplementedError('Not implemented for spark')

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
        raise NotImplementedError('Not implemented for spark')

        tmp = coldata[self.input].copy()
        tmp[np.isnan(tmp)] = self.surrogate
        coldata[self.output] = tmp

    def fit_(self, coldata_subset):
        raise NotImplementedError('Not implemented for spark')

        if self.freezed:
            return
        if self.method == 'mean':
            self.surrogate = np.nanmean(coldata_subset[self.input])
        else:
            raise Exception(f'Unknown imputation method: {self.method}')

    def model_to_json(self):
        res = {
            'Impute': {'input': self.input, 'output': self.output, 'surrogate': self.surrogate}
        }
        return res


from pyspark.ml.feature import OneHotEncoder, VectorAssembler


class Onehot(Stage, Fitable):
    len: Optional[int]
    spark_model: Any

    def fit_(self, coldata_subset):
        if self.freezed:
            return
        # print('dbg',next(iter(coldata_subset.values())).shape[0])
        encoder = OneHotEncoder(inputCol=spark_encode_colnames([self.input])[0],
                                outputCol=spark_encode_colnames([self.output])[0], dropLast=False)
        model = encoder.fit(coldata_subset.df)

        self.len = list(model._java_obj.categorySizes())[0]
        self.spark_model = model

    def transform(self, coldata: SparkColdata):
        newdf = self.spark_model.transform(coldata.df)
        coldata.setdf(newdf)

    def model_to_json(self):
        res = {
            'Onehot': {'input': self.input, 'output': self.output, 'len': self.len}
        }
        return res


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
        raise NotImplementedError('Not implemented for spark')

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
        raise NotImplementedError('Not implemented for spark')

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
        raise NotImplementedError('Not implemented for spark')
        if self.old2new is None or self.isIdentity:
            coldata[self.output] = coldata[
                self.input]  # If we don't suppose immutability this line is unsafe and might require 'copy'
        else:
            coldata[self.output] = self.old2new[compressArr(coldata[self.input])]

    def fit_(self, coldata_subset):
        raise NotImplementedError('Not implemented for spark')
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


class Pipeline(Stage, Fitable):
    stages: List[Stage]
    verbose: bool = True

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
            if isinstance(stage.output, list):
                raise Exception('Not implemented')

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
                      verbose=None):  # It should override this method because fit and transform of stages should be in order
        if verbose == None:
            verbose = self.verbose

        self.assertAllColnamesArePresent(coldata)
        for stage in self.stages:
            if verbose:
                print('eval stage:', stage)
            if isinstance(stage, Fitable):
                stage.fit_transform(coldata)
            else:
                stage.transform(coldata)

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

    def iter_stages(self):
        for stage in self.stages:
            if isinstance(stage, Pipeline):
                for ch_stage in stage.iter_stages():
                    assert (not isinstance(ch_stage, Pipeline))  # Means bug
                    yield ch_stage
            else:
                yield stage


lock = threading.Lock()


class Learner(Stage, Fitable):
    model: str
    params: Optional[dict] = {}
    label: str
    clf: Optional[Any]
    learned_model: Optional[Any]

    def getRequiredColnames(self):
        res = super().getRequiredColnames()
        res += [self.label]
        return res

    def fit_(self, coldata_subset):

        from pyspark.ml.feature import VectorAssembler

        assembler = VectorAssembler(
            inputCols=spark_encode_colnames(self.input),
            outputCol="spark_tmp_features")

        tmpdf = assembler.transform(coldata_subset.df)

        learner = self
        with lock:
            #             print(learner.model)
            clfClass = pydoc.locate(learner.model, forceload=0)

        # Xtrain, Ytrain = make_XY(coldata_subset, learner.input, learner.label)

        # data_stats = {'n_train': len(Ytrain)}
        #
        # data_stats['class_ratios'] = {'train': Ytrain.mean(axis=0).tolist()}

        y_colname = spark_encode_colnames([self.label])[0]

        # if len(np.unique(Ytrain)) > 2:
        if tmpdf.groupBy(y_colname).count().count() > 2:
            # To handle multiclass, it is required to handle the to_json and also the output differently (e.g. now just the probablity of second class is returend)
            raise Exception(
                'Currently only binray classification is supported.')  # The only problem is the json converor and which now behave in 1 dimension

        tmpdf = tmpdf.withColumn(y_colname, tmpdf[y_colname].cast(DoubleType()))

        clf = clfClass(featuresCol='spark_tmp_features',
                       labelCol=y_colname,
                       # predictionCol=spark_encode_colnames([self.output])[0],
                       probabilityCol=spark_encode_colnames([self.output])[0],
                       **learner.params)

        self.learned_model = clf.fit(tmpdf)
        self.clf = clf  # Not sure if it is needed (just kept the scikit implementation)

    def transform(self, coldata: SparkColdata):
        from pyspark.sql.functions import udf
        from pyspark.sql.types import FloatType

        assembler = VectorAssembler(
            inputCols=spark_encode_colnames(self.input),
            outputCol="spark_tmp_features")

        tmpdf = assembler.transform(coldata.df)

        tmpdf = self.learned_model.transform(tmpdf)

        output_colname = spark_encode_colnames([self.output])[0]
        secondelement = udf(lambda v: float(v[1]), FloatType())
        tmpdf = tmpdf.select(['_spark_id', secondelement(output_colname).alias(output_colname)])

        coldata.setdf(coldata.df.join(tmpdf, '_spark_id', 'outer'))

    to_json_converters = {
        'pyspark.ml.classification.LogisticRegression': lambda leaned_model:
        {
            'model': 'pyspark.ml.classification.LogisticRegression',
            'intercept': leaned_model.intercept,
            'coef': list(leaned_model.coefficientMatrix[0])
        }
    }

    def model_to_json(self):
        cls_name = self.model  # util.class_fullname(self.clf)

        if cls_name in self.to_json_converters:
            json = self.to_json_converters[cls_name](self.learned_model)
        else:
            json = {'error': f'No json convertor found for {cls_name}'}

        json.update({'input': self.input, 'output': self.output})
        return {'Learner': json}


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
