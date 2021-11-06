import sys
sys.path.append('/root/jupyter/mlaas')
from mlservice.common.util import *

print(sys.version_info)

import numbers
from collections import Set, Mapping, deque
from bson import ObjectId
from tqdm.notebook import tqdm as tqdm
import urllib
import urllib
from pymongo import MongoClient
from mlservice.common import util
from datetime import datetime

from collections import OrderedDict
import sys
import pymongo
import csv
import copy

import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics
import pandas as pd


zoneType_str2int=TapsellConstants.zoneType_str2int
zoneType_int2str=TapsellConstants.zoneType_int2str
action_str2int=TapsellConstants.action_str2int
action_int2str=TapsellConstants.action_int2str


coldata=load('tmp')


from mlservice import recipes

use_spark=True
if use_spark:
    from mlservice.stages import spark_stages as mls
else:
    from mlservice import schemas_v2 as mls



from pyspark import SparkContext
from pyspark.sql import SparkSession

import os

os.environ["SPARK_HOME"] = "/root/software/anaconda/lib/python3.7/site-packages/pyspark/"
os.environ["PYSPARK_PYTHON"]="/root/software/anaconda/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"]="/root/software/anaconda/bin/python"


spark = SparkSession.builder \
    .master('local') \
    .appName('myAppName2') \
    .config('spark.executor.memory', '50gb') \
    .config("spark.executor.instances", "15") \
    .config('spark.executor.cores', '2') \
    .config("spark.cores.max", "30") \
    .getOrCreate()
# .config('PYSPARK_PYTHON','3.7')\


print(spark.sparkContext.getConf().getAll())


input=[ 'creative-media-zone-conversion.cr']#,'campaign-media-zoneType-conversion.cvr'
label='converted'
length=100

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark.sparkContext)

# to read parquet file
df = sqlContext.read.parquet('/root/data/tapsell/parquet')

df=df.filter(df.zoneType==TapsellConstants.zoneType_str2int['REWARDED_VIDEO'])
df=df.filter(df.incomeType==3)

#
#
# pdf = pd.DataFrame(coldata)
# df = spark.createDataFrame(pdf)
print('df.count()',df.count())
df=df.repartition(100)
print('df.rdd.getNumPartitions()',df.rdd.getNumPartitions())

max_ts=df.agg({"timestamp": "max"}).first()[0]
min_ts=df.agg({"timestamp": "min"}).first()[0]
lastTrainTimestamp=(max_ts-min_ts)*0.9+min_ts
# lastTrainTimestamp=df.agg({"timestamp": "max"}).first()[0]-1000*60*60*24*24
df=df.withColumn('trainSw',df['timestamp']<lastTrainTimestamp)
df = df.na.fill(0)

coldata=mls.SparkColdata(df)
print(coldata.df.head())

quantizationLevels=8

pipeline_quantize_rates=mls.Pipeline(stages=[
    recipes.recipe_quantize_1hot(spark=use_spark, input=[col], label='converted',output=f'{col}-ready',train_sw_colname='trainSw',num_splits=quantizationLevels)
     for col in ['campaign-media-zone-conversion.impressions']
])

features=['campaign-media-zone-conversion.impressions-ready']

learner=mls.Learner(model='pyspark.ml.classification.LogisticRegression',# sklearn.linear_model.LogisticRegression',
                params={'maxIter':200},
                train_sw_colname='trainSw',
                input=features,
                output='estimatedCVR',
                label='converted')


# all_pipeline=mls.Pipeline(stages =[pipeline_quantize_rates,pipeline_onehot, learner])
all_pipeline=mls.Pipeline(stages =[pipeline_quantize_rates, learner])

tic()
all_pipeline.fit_transform(coldata)
toc()
print(coldata.df.head())
