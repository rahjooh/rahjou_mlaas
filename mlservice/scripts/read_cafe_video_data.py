import numbers
import re
import urllib
from datetime import datetime
from datetime import timedelta
import numpy as np
import scipy as sp
from bson import ObjectId
from pymongo import MongoClient
import mlservice.common.util as util


def rename_keys(dic, name_mappings):
    for oldname, newname in name_mappings:
        dic[newname] = dic[oldname]
        del dic[oldname]

    return dic


class SparseTmp():
    def __init__(self, shape=(None, None), dtype=None):
        self.rows = []
        self.cols = []
        self.vals = []
        self.dtype = dtype
        self.shape = shape

    def __setitem__(self, key, val):
        self.rows.append(key[0])
        self.cols.append(key[1])
        self.vals.append(val)

    def toSklearn(self, targetConstructor=sp.sparse.csr_matrix):
        shape = self.shape
        nrows, ncols = self.shape
        if nrows is None:
            nrows = max(self.rows + [-1]) + 1
        if ncols is None:
            ncols = max(self.cols + [-1]) + 1

        vals = np.array(self.vals)
        if len(vals) > 0:
            vals = util.compressArr(vals)
        dtype = self.dtype
        if dtype is None:
            dtype = vals.dtype

        return targetConstructor((vals, (self.rows, self.cols)), shape=(nrows, ncols), dtype=dtype)

    def tocsr(self):
        return self.toSklearn(sp.sparse.csr_matrix)


def read_mongo_to_stream(server, database, collname, query, username, password, port=27017, max_read_size=-1,
                         sort=None):
    username = urllib.parse.quote_plus(username)
    password = urllib.parse.quote_plus(password)
    mongoAuthURL = f'mongodb://{username}:{password}@{server}:{port}'

    client_tmp = MongoClient(mongoAuthURL)
    db_tmp = client_tmp.get_database(database)

    doc = db_tmp[collname].find_one(query)
    if doc is None:
        raise Exception(f'No data found to read in {collname}')

    if query == {}:
        n_expected_documents = db_tmp[collname].estimated_document_count()
        n_max = int(n_expected_documents * 1.1)
        print(f"db_tmp[{collname}].estimated_document_count() => ", n_expected_documents)
    elif (max_read_size > 0):
        n_expected_documents = max_read_size  # don't waste time on counting if max_read_size is set
        n_max = max_read_size
        print(f"n_expected_documents = max_read_size => ", n_expected_documents)
    else:
        n_expected_documents = db_tmp[collname].count(query)
        n_max = int(n_expected_documents * 1.1)
        print(f"db_tmp[{collname}].count(query) => ", n_expected_documents)

    stream = db_tmp[collname].find(query)
    if (sort):
        stream = stream.sort(sort)

    return stream, n_expected_documents, n_max


def read_stream_to_coldata(stream, n_max, n_expected_documents=None, flattened=False, rename={}, indexers=None,
                           fixed_index_columns=[], ignore_columns=[]):
    if indexers == None:
        indexers = {}

    if n_expected_documents is None:
        n_expected_documents = n_max

    tmp = {}
    coldata_extra = {}
    colnames = []
    curInd = 0
    for i, elem in enumerate(stream):
        if (i % 1000 == 0):
            util.thread_check_all()

        if (i >= n_max):
            i -= 1  # at the end i should be the last read item
            break

        if not flattened:
            elem = util.flatten(elem)

        elem = rename_keys(elem, rename)

        for col in elem.keys():
            if col in ignore_columns or col in colnames:
                continue

            if (isinstance(elem[col], numbers.Number)
                    or isinstance(elem[col], str)
                    or isinstance(elem[col], datetime)
                    or isinstance(elem[col], ObjectId)
                    or isinstance(elem[col], dict)
            ):
                if col in fixed_index_columns:
                    # The fixed_index_colnames should be first so that they will be checked later for excluding the rows
                    # (othervise extra things might get added to other columns)
                    colnames.insert(0, col)
                else:
                    colnames.append(col)

                coldata_extra[col] = np.zeros(n_max)
                coldata_extra[col].fill(np.nan)
                print('columnd detected: ', col)

            if (isinstance(elem[col], str) or isinstance(elem[col], ObjectId)) and (not col in indexers):
                indexers[col] = util.Indexer()
                print('Indexer created: ', col)

            if isinstance(elem[col], dict):
                coldata_extra[col] = SparseTmp()

        tmp.clear()
        valid_sw = True
        for colname in colnames:
            try:
                val = elem.get(colname, None)

                #             if(colname in indexers):
                #                 val=indexers[colname].add(val)
                #                 if (colname == id_colname):
                #                     val = indexers[colname].add(val)

                if isinstance(val, str) or isinstance(val, ObjectId):
                    if colname in fixed_index_columns:
                        if val in indexers[colname].key2index:
                            val = indexers[colname][val]
                        else:
                            valid_sw = False
                            break
                    else:
                        val = indexers[colname].add(val)

                if isinstance(val, datetime):
                    val = int(util.datetime2milis(val, isUTC=True))

                #                         print(colname,val)
                tmp[colname] = val
            except Exception as ex:
                print('exception on:')
                for ttt in elem:
                    print(ttt, type(elem[ttt]), elem[ttt])

                raise (ex)

        if valid_sw:
            for colname in colnames:
                val = tmp[colname]
                if val is None:
                    continue
                if not isinstance(val, dict):
                    coldata_extra[colname][curInd] = val
                else:
                    for dim, dimval in val.items():
                        if dimval > 0:
                            coldata_extra[colname][curInd, dim] = dimval
            curInd += 1

    for colname in colnames:
        if isinstance(coldata_extra[colname], SparseTmp):
            shape = coldata_extra[colname].shape
            coldata_extra[colname].shape = (curInd, shape[1])
            coldata_extra[colname] = coldata_extra[colname].tocsr()
        else:
            coldata_extra[colname] = util.compressArr(coldata_extra[colname][:curInd])

    return coldata_extra, indexers


def doc_to_standard_sparse(obj, index_name='index', res=None, path=None, index=None):
    if path is None:
        path = []
    if res is None:
        res = {}

    if isinstance(obj, dict):
        for key, val in obj.items():
            doc_to_standard_sparse(obj[key], index_name, res, path=path + [key], index=index)
    elif isinstance(obj, list):
        if not index is None:
            raise Exception('List inside list is not supported.')  # It can easilly be implemented if needed later
        for i, elem in enumerate(obj):
            if not index_name in elem:
                raise Exception('Wrong format: first elements of list should have index_name={index_name} field.')

            elem_no_index = {k: v for k, v in elem.items() if not k == index_name}
            doc_to_standard_sparse(elem_no_index, index_name, res, path=path, index=elem[index_name])
    else:
        pathStr = '.'.join(path)
        if index is None:
            res[pathStr] = obj  # copy.deepcopy(obj)
        else:
            if not pathStr in res:
                res[pathStr] = {}
            res[pathStr][index] = obj

    return res


def prepare_cafe_video_stream(mongo_stream):
    for doc in mongo_stream:
        newDoc = {}
        newDoc.update(doc_to_standard_sparse(doc['features']['requestFeature']))
        newDoc.update(doc_to_standard_sparse(doc['features']['creativeFeature']))
        newDoc['timestamp'] = doc['timestamp']
        if 'events' in doc:
            newDoc.update(doc_to_standard_sparse(doc['events'], path=['eventsTime']))
            newDoc.update(
                doc_to_standard_sparse({event: True for (event, timestamp) in doc['events'].items()}, path=['events']))
        yield newDoc


def read_yesterday_data():
    username, password = ('admin', 'cLVabiJ1IiqJYHldbfhduvadhj40UyXcLVabiJ1I')
    server = '172.16.19.201'
    port = 27017

    database = 'dataset'
    query = {}
    sort = None

    now = datetime.now()
    dayToRead = now - timedelta(days=1)
    collname = f'suggestions-{dayToRead.year}-{dayToRead.month}-{dayToRead.day}'

    mongoStream, n_expected_documents, n_max = read_mongo_to_stream(server, database, collname, query, username,
                                                                    password,
                                                                    port, sort)
    sampleStream = prepare_cafe_video_stream(mongoStream)
    coldata, indexers = read_stream_to_coldata(sampleStream, n_max, n_expected_documents, flattened=True)

    for colname in list(coldata):
        if colname.startswith('events.'):
            coldata[colname][np.isnan(coldata[colname])] = False
            coldata[colname] = util.compressArr(coldata[colname], convert_to_bool=True)

    year, month, day = re.findall('.*-(.*)-(.*)-(.*)', collname)[0]
    util.coldata_save(f'/root/data/cafe-video/training-data/daily/year={year}/month={month}/day={day}', coldata,
                      indexers)

def concatenate_recent_daily_data():
    from datetime import timedelta
    now = datetime.now()

    target_folder = '/root/data/cafe-video/training-data/recent_concatenated'

    import os
    tmp = [now - timedelta(days=i) for i in range(0, 20)]
    subfolders = [f'/root/data/cafe-video/training-data/daily/year={t.year}/month={t.month}/day={t.day}' for t in tmp]
    subfolders = [folder for folder in subfolders if os.path.isdir(folder)]

    print(subfolders)

    load_indexers = True

    indexers = util.coldata_load(subfolders[0], load_coldata=False, load_indexers=load_indexers, usetqdm=False)

    indexers

    from mlservice.common.util import coldata_load

    badColumns = set()
    if (load_indexers):
        consistent_index_cols = set(indexers.keys())
        # Load all indexers:
        for folder in subfolders[1:]:
            indexers_old = coldata_load(folder, colnames=consistent_index_cols, load_indexers=load_indexers,
                                        load_coldata=False, usetqdm=False)
            #         print(indexers_old.keys())

            for colname in list(consistent_index_cols):
                if colname in indexers_old:
                    sw_correct = True
                    for i in range(len(indexers_old[colname].index2key)):
                        if (i >= len(indexers[colname].index2key) or indexers[colname].index2key[i] !=
                                indexers_old[colname].index2key[i]):
                            print(f'Inconsistency found in indexers: {colname}')
                            sw_correct = False
                            break
                else:
                    print(f'Column not found {colname}.')
                    sw_correct = False

                if (not sw_correct):
                    #                 print(f'Indexed column {colname} was not mergable and removed.')
                    consistent_index_cols.remove(colname)
                    badColumns.add(colname)
        print('Bad columns to remove: ', badColumns)

    for colname in badColumns:
        #     if colname in coldata:
        #         del coldata[colname]
        if colname in indexers:
            del indexers[colname]

    for colname in indexers:
        util.save(f'{target_folder}/indexers/{colname}', indexers[colname], python_pickle_base='')

    timecolname = 'timestamp'

    import glob
    import os
    # print(os.path.join(folder,'columns/*.pickle'))

    folder_2_coldata_old_keep_sw = {}

    curReadTimes = coldata_load(subfolders[0], colnames=[timecolname], load_indexers=False, usetqdm=False)[timecolname]

    for folder in subfolders[1:]:
        timestampReadAgain = curReadTimes.min()  # coldata[timecolname].min()

        print('reading ', folder)
        time_file = os.path.join(folder, f'columns/{timecolname}.pickle')
        curReadTimes = util.load(time_file[:-7], python_pickle_base='')

        folder_2_coldata_old_keep_sw[folder] = curReadTimes < timestampReadAgain
        print('Data ratio to be removed as old:', 1 - folder_2_coldata_old_keep_sw[folder].mean())

    # coldata_filters=[(lambda coldata: coldata['zoneType']==TapsellConstants.zoneType_str2int['REWARDED_VIDEO'])]
    coldata_filters = [(lambda coldata: coldata['events.impression'] == True)]

    folder_2_coldata_filter_keep_sw = {}

    curReadTimes = coldata_load(subfolders[0], colnames=[timecolname], load_indexers=False, usetqdm=False)[timecolname]

    for folder in subfolders:
        timestampReadAgain = curReadTimes.min()  # coldata[timecolname].min()

        print('reading ', folder)
        coldata_tmp = util.defaultdict_keyed(
            lambda colname: util.load(os.path.join(folder, f'columns/{colname}'), python_pickle_base=''))

        coldataSw = None
        for filt in coldata_filters:
            if coldataSw is None:
                coldataSw = filt(coldata_tmp)
            else:
                coldataSw &= filt(coldata_tmp)

        #     print(np.unique(coldata_tmp['zoneType'][coldataSw]))
        folder_2_coldata_filter_keep_sw[folder] = coldataSw
        print('Data ratio to be removed:', 1 - folder_2_coldata_filter_keep_sw[folder].mean())
        del coldata_tmp  # it can even be deleted inside filters loop for more memory efficiency

    for i, folder in enumerate(subfolders):
        newcolnames = set(
            [os.path.basename(colfile)[:-7] for colfile in glob.glob(os.path.join(subfolders[0], 'columns/*.pickle'))])
        if i == 0:
            shared_colnames = set(newcolnames)
        else:
            shared_colnames.intersection_update(newcolnames)

    shared_colnames -= set(badColumns)
    print(f'reading {len(shared_colnames)} columns.')

    import traceback

    for colname in shared_colnames:
        try:
            arrs = []
            for i, folder in enumerate(subfolders):
                colfile = os.path.join(folder, f'columns/{colname}.pickle')
                #             if((colnames is None) or (colname in colnames)):
                #         coldata[colname]=load(colfile[:-7], python_pickle_base='')
                if i == 0:
                    arrs.insert(0,
                                util.load(colfile[:-7], python_pickle_base='')[folder_2_coldata_filter_keep_sw[folder]])
                else:
                    arrs.insert(0, util.load(colfile[:-7], python_pickle_base='')[
                        folder_2_coldata_filter_keep_sw[folder] & folder_2_coldata_old_keep_sw[folder]])

            if isinstance(arrs[0], sp.sparse.base.spmatrix):
                dim = max([arr.shape[1] for arr in arrs])
                for arr in arrs:
                    arr.resize(arr.shape[0], dim)
                util.save(f'{target_folder}/columns/{colname}', sp.sparse.vstack(arrs), python_pickle_base='')
            else:
                util.save(f'{target_folder}/columns/{colname}', util.compressArr(np.concatenate(arrs)),
                          python_pickle_base='')
        except KeyboardInterrupt as kiex:
            raise kiex
        except:
            traceback.print_exc()
            print(f'Error in merging {colname}. Column ignored.')
            badColumns.add(colname)

    del arrs


def prepare_training_data():
    read_yesterday_data()
    concatenate_recent_daily_data()


prepare_training_data()
