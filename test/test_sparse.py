import sys
sys.path.append('/root/jupyter/mlaas')
from util import *

print(sys.version_info)

import numbers
from collections import Set, Mapping, deque
from bson import ObjectId
from tqdm.notebook import tqdm as tqdm
import urllib
import urllib
from pymongo import MongoClient
import util
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


coldata=coldata_load('/root/data/videoBazaar/temp_cafevideo_synthesis_random_coldata')
coldata['trainSw']=np.random.rand(coldata['action'].shape[0])<0.5
coldata['converted']=coldata['action']>3

print(coldata['userAdGroupHist.search.click.count'])

from mlservice import recipes

use_spark=False
if use_spark:
    from mlservice.stages import spark_stages as mls
else:
    from mlservice.stages import sklearn_stages as mls


input=['userAdGroupHist.search.click.count']#,'campaign-media-zoneType-conversion.cvr'
label='converted'
length=100



quantizationLevels=8

pipeline_quantize_rates=mls.Pipeline(stages=[
    recipes.recipe_quantize_1hot(spark=use_spark, input=[col], label='converted',output=f'{col}-ready',train_sw_colname='trainSw',num_splits=quantizationLevels)
     for col in ['userAdGroupHist.search.click.count']
])

features=['userAdGroupHist.search.click.count-ready']

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

asd