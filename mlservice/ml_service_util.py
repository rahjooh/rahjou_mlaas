import os
import pickle
import pydoc
import threading
import traceback
import urllib
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

import sklearn
from bson import ObjectId, Binary
from pymongo import MongoClient
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from mlservice.common import util
from mlservice.stages import sklearn_jobs
from mlservice.common.util import datetime2milis, flatten, Indexer, compressArr

to_json_converters = {'sklearn.linear_model.logistic.LogisticRegression': lambda clf: {
    'model': 'sklearn.linear_model.logistic.LogisticRegression', 'intercept': clf.intercept_, 'coef': clf.coef_}}

mongoServer_ip_to_userpass = {
    '127.0.0.1:32768': (None, None),  # ('tapsellAI', 'N3JeNLPZFaU5TsBM27RynHZgwUdmCdp6'),
    '172.24.111.41': ('admin', 'S3rcHM0ng0AdDb'),  # SearchAd
    '127.0.0.1:27019': ('admin', 'rALU7VYXnyq9ugQ0ItYUIDYzEr4')  # MediaAd
}
mongoServer_ip_to_userpass.update(
    {key + ':27017': val for key, val in mongoServer_ip_to_userpass.items() if not ':' in key})

import numbers

cached_coldata = {}
cached_coldata_locks = defaultdict(threading.Lock)
# default_merge_key = 'requestId'

dbg = None


def make_coldata_shared(datasets_info: List[sklearn_jobs.DatasetDef], indexers, pre_sort_colname=None):
    """
    Read all the datasets and integrate them inside the coldata dictionary.
    """
    global cached_coldata

    # Extend (a subset of) coldata by loading collections which their names is given in  feature_colls and use the
    # id_colname to merge them together
    from tqdm import tqdm_notebook as tqdm

    merge_keys = {elem.merge_key for elem in datasets_info}
    if len(merge_keys) < 1:
        # id_colname = default_merge_key
        raise Exception('No merge key present for datasets!')
    elif len(merge_keys) > 1:
        raise Exception('More than one merge_key is not supported yet')
    else:
        id_colname = merge_keys.pop()

    all_coldata: Dict[sklearn_jobs.DatasetDef, Any] = {}  # {'': coldata}

    #     colls_cache_duration_ms={elem['collection']:elem['cache_valid_seconds'] for elem in learning_job['job_def']['datasets']}

    for d_info in datasets_info:
        with cached_coldata_locks[d_info]:  # Lock current element of cache for mult thread access
            dt = d_info.cache_valid_seconds  # colls_cache_duration_ms.get(collname,0)
            last_t = cached_coldata.get(d_info, {}).get('timestamp', 0)
            if (d_info in cached_coldata and datetime2milis(datetime.now()) - last_t < dt * 1000):
                coldata_extra = cached_coldata[d_info]['val']
            else:
                if d_info.type == 'mongo':
                    username, password = mongoServer_ip_to_userpass[d_info.server]

                    if username:
                        username = urllib.parse.quote_plus(username)
                        password = urllib.parse.quote_plus(password)
                        mongoAuthURL = f'mongodb://{username}:{password}@{d_info.server}'
                    else:
                        mongoAuthURL = f'mongodb://{d_info.server}'

                    client_tmp = MongoClient(mongoAuthURL)
                    db_tmp = client_tmp.get_database(d_info.database)
                    dbg = client_tmp

                    collname = d_info.collection

                    coldata_extra = {}
                    doc = db_tmp[collname].find_one(d_info.query)
                    if (doc is None):
                        raise Exception(f'No data found to be read in {collname}: {db_tmp}')
                    elem = flatten(doc)
                    colnames = []
                    for col in elem.keys():
                        # if (col == '_id'):
                        #     continue
                        if (isinstance(elem[col], numbers.Number) or isinstance(elem[col], str)
                                or isinstance(elem[col], datetime) or isinstance(elem[col], ObjectId)):
                            colnames.append(col)
                        if ((isinstance(elem[col], str) or isinstance(elem[col], ObjectId)) and (not col in indexers)):
                            indexers[col] = Indexer()
                    #                 colnames=list(set(elem.keys())-set(['_id']))

                    if (d_info.query == {}):
                        n = int(db_tmp[collname].estimated_document_count() * 1.1)
                        print(f"db_tmp[{collname}].estimated_document_count() => ",
                              db_tmp[collname].estimated_document_count())
                    else:
                        tmp = db_tmp[collname].count(d_info.query)
                        n = int(tmp * 1.1)
                        print(f"db_tmp[{collname}].count(d_info.filter) => ", tmp)

                    for colname in colnames:
                        coldata_extra[colname] = np.zeros(n)

                    for i, elem in tqdm(enumerate(db_tmp[collname].find(d_info.query))):
                        if (i % 1000 == 0):
                            util.thread_check_all()

                        if ((d_info.max_read_size >= 0) and (i >= d_info.max_read_size)):
                            i -= 1  # at the end i should be the last read item
                            break

                        elem = flatten(elem)

                        for colname in colnames:
                            try:
                                val = elem.get(colname, np.nan)

                                #             if(colname in indexers):
                                #                 val=indexers[colname].add(val)
                                if (colname == id_colname):
                                    val = indexers[colname].add(val)

                                if (isinstance(val, str) or isinstance(val, ObjectId)):
                                    val = indexers[colname].add(val)

                                if (isinstance(val, datetime)):
                                    val = int(datetime2milis(val))

                                #                         print(colname,val)
                                coldata_extra[colname][i] = val
                            except Exception as ex:
                                raise (ex)
                    for colname in colnames:
                        coldata_extra[colname] = compressArr(coldata_extra[colname][:i + 1])

                elif d_info.type == 'custom':
                    import pydoc
                    func = pydoc.locate(d_info.read_func)
                    coldata_extra = func(indexers, d_info.dict())
                else:
                    raise Exception(f'type not recognized:{d_info.type}')

                cached_coldata[d_info] = {'timestamp': datetime2milis(datetime.now()), 'val': coldata_extra}

        all_coldata[d_info] = coldata_extra

        # finally:
        #     pass
    #             client_tmp.close()

    d_info_list = list(all_coldata.keys())
    cur_ids_set = set(list(all_coldata[d_info_list[0]][id_colname]))
    # for collname in feature_colls:
    for d_info in d_info_list[1:]:
        cur_ids_set = cur_ids_set.intersection(set(list(all_coldata[d_info][id_colname])))

    if (len(cur_ids_set) == 0):
        raise Exception('No shared data in datasets to process.s')

    shared_ids = np.array(list(cur_ids_set))

    id2sw = np.zeros(len(indexers[id_colname]), np.bool)
    id2sw[shared_ids] = True

    print('Num of shared instances:', id2sw.sum())

    coldata_shared = {}
    for d_info, coldata_cur in all_coldata.items():
        prefix = d_info.alias
        if (prefix is None):
            prefix = d_info.collection if not d_info.collection is None else ''

        if (prefix != ''):
            prefix = prefix + '.'

        id2coldataInd = compressArr(np.arange(coldata_cur[id_colname].max() + 1))  # len(coldata_cur[id_colname])))
        #         print(id2coldataInd.shape, len(coldata_cur[id_colname]))
        id2coldataInd[coldata_cur[id_colname]] = np.arange(len(coldata_cur[id_colname]))
        indcs = id2coldataInd[shared_ids]
        for colname in coldata_cur:
            coldata_shared[prefix + colname] = coldata_cur[colname][indcs]

    # for d_info, coldata_cur in all_coldata.items():
    #     prefix = d_info.alias
    #     if (prefix is None):
    #         prefix = d_info.collection if not d_info.collection is None else ''
    #     if (prefix != ''):
    #         assert (np.all(coldata_shared[prefix + '.' + id_colname] == coldata_shared[id_colname]))
    #         del coldata_shared[prefix + '.' + id_colname]

    # print('current columns: ', list(coldata_shared.keys()))

    if (pre_sort_colname != None):
        sortIndcs = np.argsort(coldata_shared[pre_sort_colname])
        for colname in coldata_shared:
            coldata_shared[colname] = coldata_shared[colname][sortIndcs]

    return coldata_shared


def make_XY(coldata_shared, featuresColnames, labelColname):
    X = compressArr(np.column_stack(tuple(coldata_shared[colname] for colname in featuresColnames)))
    Y = coldata_shared[labelColname].astype(np.uint8)
    return X, Y


def split_train_test(X, Y, trainRatio, testRatio=None, random_permute=False):
    # trainRatio = 0.9
    if (testRatio == None):
        testRatio = 1 - trainRatio

    assert (testRatio + trainRatio <= 1)

    nTotal = X.shape[0]
    nTrain = int(trainRatio * nTotal)
    nTest = int(testRatio * nTotal)

    if (random_permute):
        randIndcs = np.random.permutation(np.arange(nTotal))
        X = X[randIndcs, :]
        Y = Y[randIndcs]

    Xtrain = X[-(nTrain + nTest):-nTest, :]
    Ytrain = Y[-(nTrain + nTest):-nTest]
    Xtest = X[-nTest:, :]
    Ytest = Y[-nTest:]

    return Xtrain, Ytrain, Xtest, Ytest


# reverse_assemble(vec, coldata)


def plot_feature_importances(vec, features_name_size):
    #     vec=clf.feature_importances_ # for classifiers like randomForest which have this field

    curInd = 0

    json_vec = {}

    for colname, dim in features_name_size:
        json_vec[colname] = vec[curInd:curInd + dim]
        curInd += dim

    # % matplotlib
    # inline

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    featureNames = []
    i = 0

    #     json_vec_keys=np.array(list(json_vec.keys()))
    #     keys_order=np.argsort(np.array([json_vec[key].sum() for key in json_vec_keys]))

    json_vec_keys = np.array([colname for colname, _ in features_name_size])
    keys_order = np.arange(len(json_vec_keys))

    # for i in range(len(sorted_feature_names))
    for featureName in json_vec_keys[keys_order]:

        values = json_vec[featureName]
        if (len(values) == 1):
            featureNames.append(f'{featureName}{len(featureNames):3}')
        else:
            #         tmp=[f'_{len(featureNames):3}']
            #             tmp=[f'|{len(featureNames):3}']
            tmp = [f'|{0:3}']
            for dim in range(len(values) - 2):
                #             tmp.append(f'|{len(tmp)+len(featureNames):4}')
                tmp.append(f'|{len(tmp):4}')

            #         tmp+=[f'‾{len(tmp)+len(featureNames):3}']
            #             tmp+=[f'|{len(tmp)+len(featureNames):3}']
            tmp += [f'|{len(tmp):4}']

            tmp[int(len(tmp) / 2)] = featureName + f'[{len(featureNames)}:{len(featureNames) + len(tmp)}]' + '  ' + tmp[
                int(len(tmp) / 2)]
            featureNames += tmp

    # % matplotlib
    # inline
    # Bring some raw data.
    values = np.concatenate(tuple([json_vec[featureName] for featureName in
                                   json_vec_keys[keys_order]]))  # [6, -16, 75, 160, 244, 260, 145, 73, 16, 4, 1]

    values_series = pd.Series(values)

    y_labels = featureNames  # [108300.0, 110540.0, 112780.0, 115020.0, 117260.0, 119500.0,
    # 121740.0, 123980.0, 126220.0, 128460.0, 130700.0]

    with lock:  # it seems that using plot is not thread safe
        # Plot the figure.
        plt.figure(figsize=(8, 20))
        ax = values_series.plot(kind='barh')
        ax.set_title('Features Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature Name')
        ax.set_yticklabels(y_labels)

        length = values.max() - values.mean()
        ax.set_xlim(values.min(), values.max() + length / 10)  # expand xlim to make labels easier to read

        rects = ax.patches

        # For each bar: Place a label
        for rect in rects:
            # Get X and Y placement of label from rect.
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: Place label left of bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label at right
                ha = 'right'

            # Use X value as label and format number with one decimal place
            label = "{:.3f}".format(x_value)

            # Create annotation
            plt.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(space, 0),  # Horizontally shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                va='center',  # Vertically center label
                ha=ha)  # Horizontally align label differently for
            # positive and negative values.


def extract_model_params(clf):
    cls_name = util.class_fullname(clf)
    if ('to_json' in dir(clf)):
        return clf.to_json()
    elif cls_name in to_json_converters:
        return to_json_converters[cls_name](clf)
    else:
        return {'error': f'{type(clf)} does not implement a to_json(self) method'}
    #     if isinstance(clf, LogisticRegression):
    #         json_model={'model':'sklearn.linear_model.logistic.LogisticRegression', 'intercept':clf.intercept_, 'coef':clf.coef_}
    #     else:
    #         return {'error':f'type not supported: {type(clf)}'}
    #         raise Exception(f'type not supported: {type(clf)}')
    return json_model


collections_last_SizeTime = defaultdict(lambda: (-1, 0))


def mongo_elapsed_time_from_last_insert(d_info: sklearn_jobs.DatasetDef, server_database=(None, None)):
    # if (server_database == (None, None)):
    #     db_tmp = db_features
    # else:
    username, password = mongoServer_ip_to_userpass[d_info.server]
    # R&D database
    username = urllib.parse.quote_plus(username)
    password = urllib.parse.quote_plus(password)
    mongoAuthURL = f'mongodb://{username}:{password}@{d_info.server}'

    client_tmp = MongoClient(mongoAuthURL)
    db_tmp = client_tmp.get_database(d_info.database)

    ## Tries to find the time passed from "last modified" of the collection
    ## Note it just starts counting milliseconds after the first it was used on a collection (first call always returns 0)
    times = []
    # for collname in feature_colls:
    collname = d_info.collection
    coll = db_tmp[collname]
    curTime = datetime2milis(datetime.now())
    (prev_size, prev_time) = collections_last_SizeTime[(server_database, collname)]
    size = coll.estimated_document_count()
    if (prev_size != size):
        collections_last_SizeTime[(server_database, collname)] = (size, curTime)
        times.append(0)
    if (prev_size == size):
        times.append(curTime - prev_time)

    # if (db_tmp != db_features):
    client_tmp.close()

    return min(times)


def create_job_for_new_collections(db_features):
    jobs_coll = db_features['_learning_jobs']
    settings_coll = db_features['_settings']

    settings_doc = settings_coll.find_one()

    for doc in settings_doc['auto_job_creation']:
        prefix = doc['collection_prefix']
        default_merge_key = doc['default_merge_key']

        #     template = doc['template']
        template = sklearn_jobs.LearningJob(**doc['template'])  # .dict() # Check format
        #     if(prefix.startswith('_')):
        #         raise(Exception('collection_prefix can not start with _ (it is reserved for settings)'))

        already_added_colls = set()
        for elem in jobs_coll.find():
            job: sklearn_jobs.LearningJob = sklearn_jobs.LearningJob(**elem)
            feature_colls = [ds.collection for ds in job.job_def.datasets]
            already_added_colls.update(feature_colls)

        for collname in db_features.list_collection_names():
            try:
                if (not collname.startswith(prefix)):
                    continue

                if (collname in already_added_colls):
                    continue

                sample = db_features[collname].find_one()
                if (sample is None):
                    continue
                extra_features = list(set(list(sample.keys())) - set(['_id', default_merge_key]))
                extra_features = [f'{collname}.' + elem for elem in extra_features]

                job_doc = template.copy()
                job_doc.status = 'wait_for_collection_init'
                job_doc.job_name = collname
                job_doc.job_def.datasets.append(
                    sklearn_jobs.DatasetDef(**{
                        'type': 'mongo',
                        'server': '127.0.0.1:27017',
                        'database': 'feature-engineering',
                        'collection': collname,
                        'merge_key': default_merge_key,
                        'cache_valid_seconds': 24 * 3600,
                        'max_read_size': 1000000}
                                            ))
                for elem in job_doc.job_def.pipeline:
                    if (isinstance(elem, sklearn_jobs.Learner)):
                        elem.input.extend(extra_features)
                # print(job_doc.dict())
                jobs_coll.insert_one(job_doc.dict())
            except Exception as ex:
                print(ex.__repr__())
                traceback.print_exc()


def coldata_get_subset(coldata, coldataSw=None, colnames=None):  # columns and rows to select
    if coldataSw is None:
        n = next(iter(coldata.values())).shape[0]
        coldataSw = np.ones(n, dtype=np.bool)
    if colnames is None:
        colnames = list(coldata.keys())

    if isinstance(colnames, str):  # if not isinstance(colnames, list):
        colnames = [colnames]
    return {colname: coldata[colname][coldataSw] for colname in colnames}


def do_preprocess_stages(json_pipeline, coldata):
    for stage in json_pipeline:
        sw = False
        #         if 'fit' in dir(stage):
        # print('dbg', isinstance(stage, schemas.Fitable), stage)
        #         if isinstance(stage,schemas.Fitable) or StageGroup:
        #             print('dbg',isinstance(stage,schemas.Fitable) , not stage.freezed)

        if isinstance(stage,
                      sklearn_jobs.Fitable):  # and not stage.freezed: # The freezed check has moved to the stage itself
            cond = stage.train_sw_colname
            coldata_sw = None
            if not cond is None:
                coldata_sw = coldata[cond]

            subcolnames = stage.input if isinstance(stage.input, list) else [stage.input]
            if 'label' in dir(stage):
                subcolnames += [stage.label]
            stage.fit(coldata_get_subset(coldata, coldata_sw, subcolnames))
            sw = True
        if 'execute' in dir(stage):
            stage.execute(coldata)
            sw = True
        if 'transform' in dir(stage):
            stage.transform(coldata)
            sw = True
        if isinstance(stage, sklearn_jobs.Learner):  # not(sw):
            yield None

    yield None


lock = threading.Lock()


def run_learning_job(learning_job, jobs_coll, indexers):
    try:
        print('job name: ', learning_job.job_name)

        #                         update_doc['$set']= {'status': 'Started', 'res': {}}
        #                         update_doc['$set']['last_modified'] = datetime.now()
        #                         jobs_coll.update_one({'_id': ID_object}, update_doc, upsert=True)
        learning_job.status = 'Started'
        learning_job.res = {}
        learning_job.last_modified = datetime.now()
        if (not jobs_coll is None):
            jobs_coll.replace_one({'_id': learning_job.id}, util.mongo_encode(learning_job.dict()))

        util.tic(learning_job.id)
        #                 id_colname=learning_job['job_def']['datasets']['merge_key']

        pre_sort_colname = learning_job.job_def.pre_sort_colname

        #               learning_job['job_def']['max_data_size']
        coldata_shared = make_coldata_shared(learning_job.job_def.datasets, indexers
                                             , pre_sort_colname=pre_sort_colname)

        stage_stream = do_preprocess_stages(learning_job.job_def.pipeline, coldata_shared)
        next(stage_stream)

        learner = [elem for elem in learning_job.job_def.pipeline if isinstance(elem, sklearn_jobs.Learner)][0]
        with lock:
            print(learner.model)
            clfClass = pydoc.locate(learner.model, forceload=0)
        clfParams = learner.params
        labelColname = learner.label
        featuresColnames = learner.input

        trainRatio = learning_job.job_def.train_ratio
        test_ratio = learning_job.job_def.test_ratio

        features_name_size = []
        for colname in featuresColnames:
            if (len(coldata_shared[colname].shape) < 2):
                dim = 1
            else:
                dim = coldata_shared[colname].shape[1]
            #     print(colname, vec[curInd:curInd+dim])
            features_name_size.append((colname, dim))

        #                         print({colname:coldata_shared[colname] for colname in featuresColnames+[labelColname]})
        X, Y = make_XY(coldata_shared, featuresColnames, labelColname)
        Xtrain, Ytrain, Xtest, Ytest = split_train_test(
            X, Y, trainRatio, test_ratio, random_permute=False)

        data_stats = {'n_train': len(Ytrain), 'n_test': len(Ytest)}
        if (pre_sort_colname != None):
            data_stats[pre_sort_colname] = {
                'min': coldata_shared[pre_sort_colname].min(),
                'max': coldata_shared[pre_sort_colname].max()
            }
        data_stats['class_ratios'] = {'train': Ytrain.mean(axis=0).tolist(), 'test': Ytest.mean(axis=0)}

        #                         del coldata_shared

        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, min_samples_leaf=512 * 16,
                                                      max_features='sqrt', n_jobs=14)

        clf.fit(Xtrain, Ytrain)
        plot_feature_importances(clf.feature_importances_, features_name_size)

        ID = learning_job.id.__str__()
        folder = f'leraning_tasks_results/{ID}/'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(folder + 'feature_importances')
        #                         update_doc['$set'] = {'status': 'Finished', 'res': {

        #                             'feature_importances_png': f'http://localhost:8888/view/Main/MLService/leraning_tasks_results/{ID}/feature_importances.png'}}
        #                         update_doc['$set']['res']['feature_importances'] = clf.feature_importances_

        learning_job.status = 'Finished'
        learning_job.res = {
            'feature_importances_png': f'http://localhost:8888/view/Main/MLService/leraning_tasks_results/{ID}/feature_importances.png',
            'feature_importances': clf.feature_importances_
        }

        #                 clf=LogisticRegression(n_jobs=14)
        clf = clfClass(**clfParams)
        clf.fit(Xtrain, Ytrain)
        coldata_shared[learner.output] = clf.predict_proba(X)

        next(stage_stream)

        learning_job.res['learned_model_pickle'] = Binary(pickle.dumps(clf))
        learning_job.res['learned_model_json'] = extract_model_params(clf)

        #                         update_doc['$set']['res']['eval_train'] = {}
        #                         Ypred = clf.predict_proba(Xtrain)
        #                         Ytrue = Ytrain

        #                         for func_name in learning_job.job_def.evaluation:
        #                             func = pydoc.locate(func_name, forceload=1)
        #                             update_doc['$set']['res']['eval_train'][func.__name__] = np.round(func(Ytrue, Ypred), 5)

        #                         update_doc['$set']['res']['eval_train']['Ypred_mean'] = Ypred.mean(axis=0).tolist()

        #                         update_doc['$set']['res']['eval_test'] = {}
        #                         Ypred = clf.predict_proba(Xtest)
        #                         Ytrue = Ytest
        #                         for func_name in learning_job.job_def.evaluation:
        #                             func = pydoc.locate(func_name, forceload=1)
        #                             update_doc['$set']['res']['eval_test'][func.__name__] = np.round(func(Ytrue, Ypred), 5)
        #                         update_doc['$set']['res']['eval_test']['Ypred_mean'] = Ypred.mean(axis=0).tolist()

        elapsed = util.toc(learning_job.id)
        learning_job.res['times'] = {'runtime_seconds': elapsed, 'finished_in': datetime.now()}
        learning_job.data_stats = data_stats
    except Exception as ex:
        #                 update_doc['$set'] = {'status': 'Error', 'res': {'error': ex.__repr__(), 'stack':traceback.format_exc()}}
        learning_job.status = 'Error'
        learning_job.res = {'error': ex.__repr__(), 'stack': traceback.format_exc()}

        prev_ex = ex
        print(ex.__repr__())
        traceback.print_exc()
    finally:
        #                 if(len(update_doc)>0):
        #                     update_doc = util.mongo_encode((update_doc)
        #                     update_doc['$set']['last_modified'] = datetime.now()
        #                     jobs_coll.update_one({'_id': ID_object}, update_doc, upsert=True)

        learning_job.last_modified = datetime.now()
        if (not jobs_coll is None):
            jobs_coll.replace_one({'_id': learning_job.id}, util.mongo_encode(learning_job.dict()))
        last_written_learning_job = learning_job.copy()


threads = {}


def process_learning_job_doc(learning_job_doc, jobs_coll, indexers, wait_time_for_insert=1000):
    learning_job_doc = util.mongo_decode(learning_job_doc)
    update_doc = {}
    try:
        learning_job: sklearn_jobs.LearningJob = sklearn_jobs.LearningJob(
            **learning_job_doc)  # munch.munchify(learning_job)
        #         print(learning_job.job_name)
        if (learning_job.schema_revision == 1):
            status = learning_job.status
            #             if('status' in learning_job):

            runSw = True
            if (status == 'Started' or status == 'Finished' or status == 'Error'):
                runSw = False

            feature_colls = [elem.collection for elem in learning_job.job_def.datasets]

            #             print(status, mongo_elapsed_time_from_last_insert(feature_colls),wait_time_for_insert)

            if status == 'wait_for_collection_init':
                wait_sw = False
                for d_info in learning_job.job_def.datasets:
                    # print('wait_for_insert',feature_colls, mongo_elapsed_time_from_last_insert(feature_colls))
                    if d_info.type == 'mongo' and mongo_elapsed_time_from_last_insert(d_info) < wait_time_for_insert:
                        wait_sw = True
                if wait_sw:
                    runSw = False

            if runSw:
                # run_learning_job(learning_job, jobs_coll, indexers)
                thread = threading.Thread(target=run_learning_job, args=(learning_job, jobs_coll, indexers))
                thread.start()
                threads[learning_job.job_name + '_' + str(thread.ident)] = thread

    except Exception as ex:
        #                 update_doc['$set'] = {'status': 'Error', 'res': {'error': ex.__repr__(), 'stack':traceback.format_exc()}}
        learning_job.status = 'Error'
        learning_job.res = {'error': ex.__repr__(), 'stack': traceback.format_exc()}

        prev_ex = ex
        print(ex.__repr__())
        traceback.print_exc()
    finally:
        #                 if(len(update_doc)>0):
        #                     update_doc = util.mongo_encode((update_doc)
        #                     update_doc['$set']['last_modified'] = datetime.now()
        #                     jobs_coll.update_one({'_id': ID_object}, update_doc, upsert=True)

        learning_job.last_modified = datetime.now()
        if not jobs_coll is None:
            jobs_coll.replace_one({'_id': learning_job.id}, util.mongo_encode(learning_job.dict()))
        last_written_learning_job = learning_job.copy()
