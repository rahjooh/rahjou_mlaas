"""
Some general utils to do sequential processes with high speed
"""

from typing import List

import pyximport
import numpy as np

pyximport.install(setup_args={"a": '', "include_dirs": np.get_include()},
                  reload_support=True, inplace=True)

import cython_modules.util as cyutil
from mlservice.common.util import compressArr, coldata_lexsort


def sequence_process(resetSig, incSig, timestamps, time_window_size):
    resetSig = compressArr(resetSig, convert_to_bool=True)
    incSig = compressArr(incSig, convert_to_bool=True)

    if (not resetSig.dtype == np.bool or not incSig.dtype == np.bool):
        raise Exception('incSig and resetSig should have boolean type.')

    return cyutil.cython_sequence_process(resetSig.astype(np.uint8), incSig.astype(np.uint8), timestamps,
                                          time_window_size)


# @noglobal
def count_in_batch(coldata, batch_key_colnames, pre_sort_colname=None, event_colname=None, time_colname=None,
                   time_window_size=None, sort_inplace=True):
    """
    This method works on coldata arrays and produce a new column which each row contains the
    number of events BEFORE that row in the "batch" of that row.
    The batch of a row i is the set of rows which satisfy these conditions:
    1. They are before the current row (after sort based on sort_colname if given)
    2. The values of columns given in batch_key_colnames are the same as row i
    3. The value of column named time_colname is LESS than or equal row i and differs by atmost time_window_size (inclusive)
    """

    if not sort_inplace:
        raise Exception(
            'Not supported yet')  # The problem is that at the end the result should be chnaged back to the initial order

    n = coldata[batch_key_colnames[0]].shape[0]

    if time_window_size is None:
        time_window_size = int(coldata[time_colname].max() - coldata[time_colname].min() + 1)

    sort_colnames: List[str] = batch_key_colnames.copy()

    if time_colname is not None:
        sort_colnames = [time_colname] + sort_colnames
    else:
        time_colname = pre_sort_colname

    if pre_sort_colname is not None:
        sort_colnames = [pre_sort_colname] + sort_colnames

    sort_cols = [coldata[colname] for colname in sort_colnames]
    coldata = coldata_lexsort(coldata, sort_cols, inplace=sort_inplace)

    if event_colname is not None:
        incSig = coldata[event_colname]
    else:
        incSig = np.ones(n, dtype=np.uint8)
    if time_colname is None:
        timestamps = np.zeros(n, dtype=np.uint8)
    else:
        timestamps = coldata[time_colname]

    indicesArrays = [coldata[colname] for colname in batch_key_colnames]
    resetSig = np.logical_or.reduce([np.ediff1d(arr, to_begin=1) > 0 for arr in indicesArrays])
    event_counts_in_group, _ = sequence_process(resetSig, incSig, timestamps, time_window_size)

    return event_counts_in_group


def freq_in_batch(coldata, batch_key_colnames, pre_sort_colname=None, event_colname=None,
                  time_colname=None,
                  time_window_size=None, sort_inplace=True, return_batch_size=False):
    """
    Like count_in_batch but it calculates the frequency of event in the batch (and not count)

    :param return_batch_size: if True,  the total number of elements in batch is also returned
    """

    batch_size = count_in_batch(coldata, batch_key_colnames, pre_sort_colname, event_colname=None,
                                time_colname=time_colname,
                                time_window_size=time_window_size, sort_inplace=sort_inplace)

    counts = count_in_batch(coldata=coldata, batch_key_colnames=batch_key_colnames,
                            pre_sort_colname=pre_sort_colname, event_colname=event_colname,
                            time_colname=time_colname,
                            time_window_size=time_window_size, sort_inplace=sort_inplace)

    res = np.zeros_like(counts, dtype=np.float)
    res[batch_size != 0] = counts[batch_size != 0] / batch_size[batch_size != 0]

    if return_batch_size:
        return res, batch_size
    else:
        return res
