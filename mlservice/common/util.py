# In jupyter add the following for auto reload  (Note: for "import *" to get reevaluated when you add a new function, you need to run it again)
# %reload_ext autoreload
# %autoreload 1
# %aimport hhk.util
# from hhk.util import *
import pickle
import io
import numpy as np
import numpy.random as random
import scipy.io as sio
import os

# import numpy as np;
import sys;
from collections import defaultdict
import collections

startTime_for_tictoc = {}


class TapsellConstants:
    zoneType_str2int = {
        'REWARDED_VIDEO': 1,
        'INTERSTITIAL_WEB_VIEW': 2,
        'INTERSTITIAL_VIDEO': 3,
        'PREROLL': 4,
        'NATIVE_VIDEO': 5,
        'NATIVE_BANNER': 6,
        'STANDARD_BANNER': 7,
        'EXPANDABLE_BANNER': 8,
        'WEBSITE_BANNER': 9,
        'IFRAME': 10}
    zoneType_int2str = {val: key for key, val in zoneType_str2int.items()}
    action_str2int = {'isActionDone': 6, 'isInstalled': 5, 'isClicked': 4, 'isCompletelyViewed': 3,
                      'isSkipPointPassed': 2, 'isImpressed': 1}
    action_int2str = {val: key for key, val in action_str2int.items()}


def tic(identifier=None):
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc[identifier] = time.time()


def toc(identifier=None, print_res: bool = True):
    import time
    global startTime_for_tictoc
    if identifier in startTime_for_tictoc:
        # if 'startTime_for_tictoc' in globals():
        if print_res:
            print("Elapsed time is " + str(time.time() - startTime_for_tictoc[identifier]) + " seconds.")
        return time.time() - startTime_for_tictoc[identifier]
    else:
        print(f"Toc: start time not set for identifier {identifier}")
        return None


default_python_pickle_base = '/root/data/hassan/python_pickle_base/';


# default_python_pickle_base="F:\\python_pickle_base\\";
def save(fileName, variable, pickle_protocol=4, python_pickle_base=None, backup=False):
    import pathlib
    import os
    if (python_pickle_base == None):
        python_pickle_base = default_python_pickle_base;

    filePathName = python_pickle_base + fileName + '.pickle'
    pathlib.Path(os.path.dirname(filePathName)).mkdir(parents=True, exist_ok=True)
    if (backup and os.path.isfile(filePathName)):
        os.replace(os.path.realpath(filePathName), os.path.realpath(filePathName) + ".bak")
    with open(filePathName, 'wb') as f:
        pickle.dump(variable, f, protocol=pickle_protocol);


def load(fileName, python_pickle_base=None):
    if (python_pickle_base == None):
        python_pickle_base = default_python_pickle_base;
    with open(python_pickle_base + fileName + '.pickle', 'rb') as f:
        return pickle.load(f);


def saveJson(fileName, variable, python_pickle_base=None):
    import ujson;
    if (python_pickle_base == None):
        python_pickle_base = default_python_pickle_base;
    with io.open(python_pickle_base + fileName + '.json', 'w', encoding='utf8') as f:
        ujson.dump(variable, f)


def loadJson(fileName, python_pickle_base=None, extension=".json"):
    import ujson;
    if (python_pickle_base == None):
        python_pickle_base = default_python_pickle_base;
    with io.open(python_pickle_base + fileName + extension, 'r', encoding='utf8') as f:
        return ujson.load(f);


class Minner:
    m = float('inf');

    def __init__(self):
        self.m = float('inf')

    def add(self, value):
        if (self.m > value):
            self.m = value;
            return 1;
        return 0;

    def value(self):
        return self.m;


class Maxer:
    m = float('-inf')

    def __init__(self):
        self.m = float('-inf');

    def add(self, value):
        if (self.m < value):
            self.m = value;
            return 1;
        return 0;

    def value(self):
        return self.m;


class Meaner:
    s = 0.0
    n = 0.0

    def __init__(self):
        self.m = 0;

    def add(self, value):
        self.s += value
        self.n += 1

    def value(self):
        return self.s / self.n;


class Variancer():
    def __init__(self):
        self.meanX = Meaner();
        self.meanX2 = Meaner();

    def add(self, value):
        self.meanX.add(value)
        self.meanX2.add(value ** 2)

    def value(self):
        return self.meanX2.value() - (self.meanX.value()) ** 2


class GroupReducer:
    def __init__(self, *reducers):
        self.reducers = reducers

    def add(self, value):
        for reducer in self.reducers:
            reducer.add(value)

    def value(self):
        return tuple(reducer.value() for reducer in self.reducers)


class Indexer:
    #     key2index=dict();
    #     index2key=list();

    def __init__(self, initIndex2key=None):
        if (initIndex2key is None):
            initIndex2key = list()
        self.index2key = initIndex2key;
        self.key2index = {key: index for index, key in enumerate(self.index2key)};
        self.key2index.pop(None,
                           None)  # Remove the None value which might be added initIndex2key (None is valid in index2key but not in key2index)

    #     def __init__(self):
    #         self.key2index=dict();
    #         self.index2key=list();

    def add(self, key):
        if (key == None):
            raise Exception('None is not a valid key')
        proposedIndex = len(self.index2key);
        index = self.key2index.setdefault(key, proposedIndex);
        if (index == proposedIndex):  # the key was new
            self.index2key.append(key);
        return index

    def add_at(self, index, key):
        if (key == None):
            raise Exception('None is not a valid key')
        prevInd = self.key2index.get(key, -1)
        prevKeySw = False
        if (len(self.index2key) > index):
            prevKeySw = True
            prevKey = self.index2key[index]
        if ((prevInd != -1 and prevInd != index) or (prevKeySw and prevKey != key and prevKey != None)):
            raise Exception(f'Inconsistent index,key:  prevInd={prevInd},prevKey={prevKey}, index={index}, key={key}')

        if (prevInd != -1):
            return False;  # No change
        else:
            curLength = len(self.index2key)
            #             if(curLength>index):
            #                 prevKey=self.index2key[index]
            #                 raise Exception(f'inconsistent index prevKey={prevKey}, index={index}, key={key}')

            if (curLength <= index):
                self.index2key.extend([None] * (index + 1 - curLength))
                if (
                        curLength < index):  # Note: at this point it should be that curLength<=index unless it the value in index2key was None for this index
                    print('warning: None values added to indexer by add_at function')

            self.index2key[index] = key
            self.key2index[key] = index

    def merge(self, otherIndexer, dtype=np.uint32):
        ind_old2new = np.zeros(len(otherIndexer.index2key), dtype=dtype)

        for oldInd, key in enumerate(otherIndexer.index2key):
            newInd = self.add(key)
            ind_old2new[oldInd] = newInd
        return ind_old2new

    def __getitem__(self, key):
        return self.key2index[key];

    def rev(self, index):  # reverse index
        return self.index2key[index]

    def __len__(self):
        return len(self.key2index.keys())


#     def __repr__(self):
#         return str(self.wikiId)+",tf=" + str(self.tf)

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def tf_reset():
    import tensorflow as tf
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    prevSess = tf.get_default_session()
    if (prevSess):
        prevSess.close()
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    return sess;


# Get size of object (copied from stackoverflow site).
import sys
from numbers import Number
from collections import Set, Mapping, deque

try:  # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError:  # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""

    def inner(obj, _seen_ids=set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)


import threading;


def thread_interrupt_check():
    t = threading.currentThread()
    return getattr(t, "interrupt", False)


def thread_pause_check():
    t = threading.currentThread()
    if (getattr(t, "pause", False)):
        print('thread paused')
        t.wakeSignal.wait()
        print('continuing thread execution...')


def thread_check_all():
    thread_pause_check()  # This should be before thread_interrupt_check because of how thread_interrupt function acts: first set thread.interrupt=True and then wakes up the thread
    if (thread_interrupt_check()):
        raise Exception('Thread interrupted')


def thread_interrupt(thread, wakeUp=True):
    print('thread.interrupt=True')
    thread.interrupt = True;
    if (wakeUp):
        thread_unpause(thread)  # Interupt will make the thread wake up if its sleep and then interrupt properly


def thread_pause(thread):
    thread.wakeSignal = getattr(thread, "wakeSignal", threading.Event())
    thread.wakeSignal.clear();
    thread.pause = True;


def thread_unpause(thread):
    wakeSignal = getattr(thread, "wakeSignal", threading.Event())
    thread.pause = False;
    wakeSignal.set();


def thread_status(thread):
    if (getattr(thread, "wakeSignal", None) and not thread.wakeSignal.isSet()):
        return 'paused'
    if (not thread.is_alive()):
        return 'interrupted'
    return 'running'


def beep(frequency=2500, duration_ms=500, times=1, delay_ms=500):
    import winsound
    import time

    for i in range(times):
        winsound.Beep(frequency, duration_ms)
        if (i != times - 1):
            time.sleep(delay_ms / 1000)


def read_dataset(name, shuffle=True):
    if (name == 'HIGGS'):
        if os.name == 'nt':
            higgsData = sio.loadmat('D:\\Data\\phd\\Code\\datasets\\HIGGS.csv\\HIGGS_matlab_100000_train_subsample.mat')
        else:
            higgsData = sio.loadmat('/home/hassan/Desktop/HIGGS_matlab_100000_train_subsample.mat')

        trainX = higgsData['trainX'];
        trainY = higgsData['trainY'];
        testX = higgsData['testX'];
        testY = higgsData['testY'];

        initMean = trainX.mean(axis=0);
        initStd = trainX.std(axis=0);
        trainX = trainX - initMean;
        trainX = trainX / initStd;
        testX = testX - initMean;
        testX = testX / initStd;

    elif (name == 'MNIST'):
        if os.name == 'nt':
            data = sio.loadmat('D:\\Data\\phd\\Code\\datasets\\MNIST\\MNIST_data')
        else:
            data = sio.loadmat('/home/hassan/Desktop/MNIST_data.mat')
        trainX = data['trainX'];
        trainY = data['trainY'];
        testX = data['testX'];
        testY = data['testY'];

    elif (
            name == 'MNIST_centered'):  # The default dataset in the mnist site. It is 28*28 instead of 20*20. It is generated by moving the center of mass in the center of the image.
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
        trainX = mnist.train.images
        trainY = mnist.train.labels
        testX = mnist.test.images
        testY = mnist.test.labels
        # Change the shape from (n,) to (n,1)
        trainY = np.expand_dims(trainY, axis=1)
        testY = np.expand_dims(testY, axis=1)


    elif (name == 'SUSY'):
        if os.name == 'nt':
            data = sio.loadmat('D:\\Data\\phd\\Code\\datasets\\SUSY\\SUSY_matlab_100000_train_subsample.mat')
        else:
            data = sio.loadmat('/home/hassan/Desktop/SUSY_matlab_100000_train_subsample.mat')
        trainX = data['trainX'];
        trainY = data['trainY'];
        testX = data['testX'];
        testY = data['testY'];

        initMean = trainX.mean(axis=0);
        initStd = trainX.std(axis=0);
        trainX = trainX - initMean;
        trainX = trainX / initStd;
        testX = testX - initMean;
        testX = testX / initStd;
    else:
        raise Exception('Dataset not known.')

    if (shuffle):
        perm = random.permutation(trainX.shape[0]);
        trainX = trainX[perm, :];
        trainY = trainY[perm, :];
        perm = random.permutation(testX.shape[0]);
        testX = testX[perm]
        testY = testY[perm]

    return (trainX, trainY, testX, testY)


def kernel_matrix(kernel_type, X, sigma, Xm):
    if (kernel_type == 'Gaussian'):
        import sklearn.metrics.pairwise
        pairwise_dists = sklearn.metrics.pairwise.pairwise_distances(X, Xm)
        K = np.exp(-(pairwise_dists ** 2) / (2 * (sigma ** 2)))
        return K;
    else:
        raise Exception('Not a valid kernel: ' + kernel_type)


def toOneOfKCoding(dataLabels):
    labelsUnique = np.unique(dataLabels)
    res = np.zeros((len(dataLabels), len(labelsUnique)), dtype=np.uint8)
    for i_label, label in enumerate(labelsUnique):
        res[:, i_label] = (dataLabels == label).transpose().astype(np.int)  # [:,0]

    return res


def oneofk(Y):  # TODO: merge with above func
    if (Y.dtype == np.bool):
        Y = Y.astype(np.uint8)
    assert (Y.min() >= 0)
    Y_one_of_k = np.zeros((Y.shape[0], Y.max() + 1))
    Y_one_of_k[np.arange(Y.shape[0]), Y] = 1
    return compressArr(Y_one_of_k)


def oneofk_reverse(Y):
    return compressArr(np.argmax(Y, axis=1))


def whatismyip():
    import requests
    ip = requests.get('https://api.ipify.org').text
    return ip


def getScopeFunctionCodes(globalsArg=globals()):
    import inspect
    res = {}
    for key, val in globalsArg.items():
        if (str(type(val)) == "<class 'function'>"):
            res[key] = inspect.getsource(val)
    return res


def getRunInfo(logScopeFunctionCodes=True, logWhatismyip=True):
    import datetime
    import sys
    import os
    import getpass
    import socket

    os.path.abspath(os.curdir)

    #     runInfo['git_hash_string']=git_hash_string;
    runInfo = {}
    runInfo['time'] = str(datetime.datetime.now())
    runInfo['utctime'] = str(datetime.datetime.utcnow())
    runInfo['sys.argv'] = sys.argv;
    runInfo['curdir'] = os.path.abspath(os.curdir)
    runInfo['username'] = getpass.getuser()
    runInfo['computername'] = os.environ['COMPUTERNAME']
    runInfo['hostname'] = socket.gethostname()
    runInfo['sys.version'] = sys.version

    if (logWhatismyip):
        runInfo['whatismyip'] = whatismyip()
    if (logScopeFunctionCodes):
        runInfo['scopeFunctionCodes'] = getScopeFunctionCodes()
    runInfo['socket.gethostbyname_ex(socket.gethostname())'] = socket.gethostbyname_ex(socket.gethostname())

    return runInfo

    def getBatches(dataX, dataY, nBatch):
        #     nBatch=nBatch+(nTrain%nBatch)//(nTrain//nBatch)

        n = dataX.shape[0]
        for i_batch in range(n // nBatch):
            indices = np.arange(i_batch * nBatch, (i_batch + 1) * nBatch)
            yield dataX[indices, :], dataY[indices, :]
        if (indices[-1] + 1 < n):
            yield dataX[indices[-1] + 1:, :], dataY[indices[-1] + 1:, :]


# Test:
# x=np.zeros([5,3])
# for batch in getBatches(x,x,2):
# print(batch)


import os
import signal
import sys
import time
from urllib.parse import urljoin
import requests


class NbServer:
    pid = port = url = notebook_dir = None

    def __init__(self, info):
        self.serverInfo = info
        self.pid = info['pid']
        self.port = info['port']
        self.url = info['url']
        self.notebook_dir = info['notebook_dir']
        self.token = info.get('token', '')

        self.last_sessions = []

    @classmethod
    def findall(cls):
        from notebook.notebookapp import list_running_servers
        return [cls(info) for info in list_running_servers()]

    @classmethod
    def find_new_and_stopped(cls, last_servers):
        last_by_pid = {s.pid: s for s in last_servers}
        new_servers, kept_servers = [], []
        for server in cls.findall():
            if server.pid in last_by_pid:
                kept_servers.append(last_by_pid.pop(server.pid))
            else:
                new_servers.append(server)

        return list(last_by_pid.values()), new_servers, kept_servers

    def check_alive(self):
        from notebook.utils import check_pid
        if not check_pid(self.pid):
            return False

        try:
            requests.head(self.url)
            return True
        except requests.ConnectionError:
            return False

    def sessions(self, password=None):
        params = {}
        if self.token:
            params['token'] = self.token

        try:
            s = requests.Session()
            resp = s.get(urljoin(self.url, 'login'))
            xsrf_cookie = resp.cookies['_xsrf']

            if (password):
                params = {'_xsrf': xsrf_cookie, 'password': password}
            else:
                params = {'_xsrf': xsrf_cookie}
            s.post(urljoin(self.url, 'login'), data=params)

            r = s.get(urljoin(self.url, 'api/sessions'), params=params)
        except requests.ConnectionError:
            self.last_sessions = []
        else:
            r.raise_for_status()
            self.last_sessions = r.json()
        return self.last_sessions

    # HHK: Testing some codes myself:
    def kernels(self, password=None):
        params = {}
        if self.token:
            params['token'] = self.token

        try:
            s = requests.Session()
            resp = s.get(urljoin(self.url, 'login'))
            xsrf_cookie = resp.cookies['_xsrf']

            if (password):
                params = {'_xsrf': xsrf_cookie, 'password': password}
            else:
                params = {'_xsrf': xsrf_cookie}
            s.post(urljoin(self.url, 'login'), data=params)

            r = s.get(urljoin(self.url, 'api/kernels'), params=params)
        except requests.ConnectionError:
            res = []
        else:
            res = r.json()
        return res

    def sessions_new_and_stopped(self, password=None):
        last_by_sid = {s['id']: s for s in self.last_sessions}
        new_sessions, kept_sessions = [], []
        for curr_sess in self.sessions(password):
            sid = curr_sess['id']
            if sid in last_by_sid:
                del last_by_sid[sid]
                kept_sessions.append(curr_sess)
            else:
                new_sessions.append(curr_sess)

        return list(last_by_sid.values()), new_sessions, kept_sessions

    def shutdown(self, wait=True):
        os.kill(self.pid, signal.SIGTERM)

        if wait:
            self.wait()

    def wait(self, interval=0.01):
        from notebook.utils import check_pid
        # os.waitpid() only works with child processes, so we need a busy loop
        pid = self.pid
        while check_pid(pid):
            time.sleep(interval)

    def stop_session(self, sid):
        r = requests.delete(urljoin(self.url, 'api/sessions/%s' % sid))
        r.raise_for_status()


def launch_server(directory, **kwargs):
    import subprocess
    cmd = [sys.executable, '-m', 'jupyterlab', directory, '--no-browser']
    if sys.platform == 'darwin' and not sys.stdin.isatty():
        script = 'tell application "Terminal" to do script "{}; exit"'.format(
            ' '.join(cmd))
        subprocess.Popen(["osascript", "-e", script])
    else:
        subprocess.Popen(cmd)


def jupyter_kernelid_to_file(kernel_id):
    import ujson

    with open(find_in_parent_folders('jupyter_info.json'), 'r') as f:
        info = ujson.loads(f.read())

    password = info['password']
    for server in NbServer.findall():
        for session in server.sessions(password):
            if (kernel_id == session['kernel']['id']):
                file = server.notebook_dir + '/' + session['notebook']['path']
                return file

    return None


def jupyter_find_notebook_dir():
    import ujson

    kernel_id = jupyter_find_kernelid()
    with open(find_in_parent_folders('jupyter_info.json'), 'r') as f:
        info = ujson.loads(f.read())

    password = info['password']
    for server in NbServer.findall():
        for session in server.sessions(password):
            if (kernel_id == session['kernel']['id']):
                return server.notebook_dir

    return None


def jupyter_find_kernelid():
    from IPython.lib.kernel import get_connection_file
    import re
    return re.findall('kernel-(.*).json', get_connection_file())[0]


def jupyter_find_file():
    return jupyter_kernelid_to_file(jupyter_find_kernelid())


def find_in_parent_folders(filename, verbose=False):
    import os
    folder = os.getcwd()
    while True:

        file = folder + '/' + filename

        parent, _ = os.path.split(folder)
        if (parent == folder):
            break
        folder = parent

        if (verbose):
            print(file, os.path.exists(file))

        if (os.path.exists(file)):
            return file

    return None


def jupyter_get_all_kernels_info(password=None):
    import re
    import string
    import psutil
    import pwd
    import ujson

    if (password == None):
        with open(find_in_parent_folders('jupyter_info.json'), 'r') as f:
            info = ujson.loads(f.read())
        password = info['password']

    kernelId2info = {}

    for server in NbServer.findall():
        for session in server.sessions(password):
            kernelId2info[session['kernel']['id']] = session
            kernelId2info[session['kernel']['id']]['serverInfo'] = server.serverInfo

    regex = re.compile(r'.+kernel-(.+)\.json')
    port_regex = re.compile(r'port=(\d+)')

    pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]

    # memory info from psutil.Process
    df_mem = []
    ports = []
    default_port = 8888

    for pid in pids:
        try:
            ret = open(os.path.join('/proc', pid, 'cmdline'), 'rt').read()
        except IOError:  # proc has already terminated
            continue

        # jupyter notebook processes
        if len(ret) > 0 and ('jupyter-notebook' in ret or 'ipython notebook' in ret):
            port_match = re.search(port_regex, ret)
            if port_match:
                port = port_match.group(1)
                ports.append(int(port))
            else:
                ports.append(default_port)
                default_port += 1
        if len(ret) > 0 and ('jupyter' in ret or 'ipython' in ret) and 'kernel' in ret:
            # kernel
            kernel_ID = re.sub(regex, r'\1', ret)
            kernel_ID = "".join(filter(lambda x: x in string.printable, kernel_ID))

            # memory
            process = psutil.Process(int(pid))
            mem = process.memory_info()[0] / float(1e9)
            creation_time = psutil.Process(int(pid)).create_time()

            # user name for pid
            for ln in open('/proc/{0}/status'.format(int(pid))):
                if ln.startswith('Uid:'):
                    uid = int(ln.split()[1])
                    uname = pwd.getpwuid(uid).pw_name

            # user, pid, memory, kernel_ID
            df_mem.append([uname, pid, mem, kernel_ID])
            kernelId2info[kernel_ID]['pid'] = pid
            kernelId2info[kernel_ID]['uname'] = uname
            kernelId2info[kernel_ID]['mem'] = mem
            kernelId2info[kernel_ID]['creation_time_epoch_ms'] = creation_time * 1000
            kernelId2info[kernel_ID]['creation_time_str'] = time.strftime("%Y-%m-%d %H:%M:%S",
                                                                          time.localtime(creation_time))

    return kernelId2info


def jupyter_get_root_folder():
    return jupyter_get_all_kernels_info()[jupyter_find_kernelid()]['serverInfo']['notebook_dir']


def jupyter_find_notebook(filename):
    import glob2
    root = jupyter_get_root_folder()
    return glob2.glob(root + '/' + '**/' + filename)


def register_dependency(dependencies):
    from IPython.lib.kernel import get_connection_file
    import re
    import ujson

    if (not isinstance(dependencies, list)):
        dependencies = [dependencies]

    tmp_abs_dep = []
    for dep in dependencies:
        if ('/' in dep):
            tmp_abs_dep.append(dep)
            continue

        if (not dep.endswith('.ipynb')):
            dep = dep + '.ipynb'
        abs_path = jupyter_find_notebook(dep)
        if (len(abs_path) == 0):
            raise Exception(f'Dependency not found: {dep}')
        if (len(abs_path) > 1):
            raise Exception(f'Ambiguous dependency {dep}: {abs_path}')
        tmp_abs_dep.append(abs_path[0])

    dependencies = tmp_abs_dep

    file_path = find_in_parent_folders('kernel_dependencies.json')
    if (file_path == None):
        raise Exception('No kernel_dependencies.json found')

    with open(file_path, 'r') as f:
        try:
            alldep = ujson.loads(f.read())
        except:
            alldep = {}

    kernelfile = get_connection_file()
    kernel = re.findall('kernel-(.*).json', kernelfile)[0]

    info = alldep.get(kernel, {})
    deps = info.get('dependencies', [])
    deps = list(set(deps + dependencies))
    info['dependencies'] = deps
    info['notebook_file'] = jupyter_find_file()
    info['connection_file'] = kernelfile

    alldep[kernel] = info

    with open(file_path, 'w') as f:
        f.write(ujson.dumps(alldep))


def unregister_dependency(dependencies):
    from IPython.lib.kernel import get_connection_file
    import re
    import ujson

    from IPython.lib.kernel import get_connection_file
    if (not isinstance(dependencies, list)):
        dependencies = [dependencies]

    tmp_abs_dep = []
    for dep in dependencies:
        if ('/' in dep):
            tmp_abs_dep.append(dep)
            continue

        if (not dep.endswith('.ipynb')):
            dep = dep + '.ipynb'
        abs_path = jupyter_find_notebook(dep)
        if (len(abs_path) == 0):
            raise Exception(f'Dependency not found: {dep}')
        if (len(abs_path) > 1):
            raise Exception(f'Ambiguous dependency {dep}: {abs_path}')
        tmp_abs_dep.append(abs_path[0])

    dependencies = tmp_abs_dep

    file_path = find_in_parent_folders('kernel_dependencies.json')
    if (file_path == None):
        raise Exception('No kernel_dependencies.json found')

    with open(file_path, 'r') as f:
        try:
            alldep = ujson.loads(f.read())
        except:
            alldep = {}

    kernelfile = get_connection_file()
    kernel = re.findall('kernel-(.*).json', kernelfile)[0]

    info = alldep.get(kernel, {})
    deps = info.get('dependencies', [])
    deps = list(set(deps) - set(dependencies))
    info['dependencies'] = deps
    #     info['notebook_file']=jupyter_find_file()
    #     info['connection_file']=kernelfile

    alldep[kernel] = info

    #     prev=alldep.get(kernel, [])

    #     prev=list(set(prev)-set(dependencies))
    #     alldep[kernel]=prev

    with open(file_path, 'w') as f:
        f.write(ujson.dumps(alldep))


import time

from datetime import datetime as datetime

_epoch = datetime.utcfromtimestamp(0)


def datetime2milis(dt, isUTC):
    """
    Note: if dt is 'naive' it does not know what timezone it is in! 
    Default behavior of pymongo needs isUTC=True
    """
    if isUTC:
        return int((dt - _epoch).total_seconds() * 1000)
    else:
        return dt.timestamp() * 1000


# Deprecated as it caused problems when input datetime is in UTC
# def datetime2milis(t=None):
#     if (t):
#         return t.timestamp() * 1000

#     return int(round(time.time() * 1000))


from datetime import timedelta

import socket
import struct


def ip2int(addr):
    try:
        return struct.unpack("!I", socket.inet_aton(addr))[0]
    except Exception as err:
        print('Error converting ip ', addr, ': ', err, )
        return 0


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


## Numpy util functions

def compressArr(arr, convert_to_bool=False, accepted_error=0):
    import scipy.sparse

    if isinstance(arr, scipy.sparse.spmatrix):
        return arr

    if arr.dtype == np.bool:  # Smallest possible typ (which I had to exlude because of some technicalities...)
        return arr

    ftypes = [np.float16, np.float32, np.float64]
    if os.name != 'nt':  # numpy.float128 is not defined in windows
        ftypes.append(np.float128)

    sw_implicit_int = np.all(arr % 1 == 0)

    if accepted_error != 0 and not sw_implicit_int and arr.dtype in ftypes:
        if accepted_error >= 1:  # this results in implicitly having int type
            arr = np.round(arr, 1)  # np.round(arr,int(np.ceil(np.log10(1/accepted_error))))
            sw_implicit_int = True
        else:
            for typ in ftypes:
                if np.abs(arr - arr.astype(typ)).max() <= accepted_error:
                    arr = arr.astype(typ)
                    sw_implicit_int = np.all(arr % 1 == 0)
                    break

    types = [np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
    if convert_to_bool:
        types = [np.bool] + types

    if arr.dtype not in types and not sw_implicit_int:  # The second condition checks for float arrays which are implicitly int
        #         raise Exception('Unsupportd type')
        return arr

    if arr.shape[0] == 0:
        return arr

    m = arr.min()
    M = arr.max()
    for typ in types:
        if typ == np.bool:
            if np.all((arr == 0) | (arr == 1)):
                break
        elif m >= np.iinfo(typ).min and M <= np.iinfo(typ).max:
            break

    if (arr.dtype == typ):
        return arr

    res = np.zeros_like(arr, dtype=typ)
    res[:] = arr[:]
    return res


def is_lexsorted(arrs):
    N = arrs[0].shape[0]
    ignore_comp_sw = np.zeros((N - 1), dtype=np.bool)

    res = True

    for arr in arrs[::-1]:
        if not np.all(ignore_comp_sw | (arr[1:] >= arr[:-1])):
            res = False
            break
        ignore_comp_sw |= arr[1:] != arr[:-1]

    return res


def coldata_from_pandas(df):
    res = {}
    for colname in df.keys():
        res[colname] = compressArr(df[colname].to_numpy())
    return res


def coldata_lexsort(coldata_shared, sort_cols, inplace=True, verbose=False):
    if not is_lexsorted(sort_cols):
        sortIndcs = np.lexsort(sort_cols)
        if verbose:
            print('Finished sorting')
        if inplace:
            for colname in coldata_shared.keys():
                coldata_shared[colname][:] = coldata_shared[colname][sortIndcs]
        else:
            coldata_shared = coldata_shared.copy()
            for colname in coldata_shared.keys():
                coldata_shared[colname] = coldata_shared[colname][sortIndcs]
    else:
        if (verbose):
            print('Already sorted')
    return coldata_shared


def coldata_save(folder, coldata, indexers=None, usetqdm=False):
    if usetqdm:
        from tqdm.notebook import tqdm

    if not indexers is None:
        for colname in indexers:
            save(f'{folder}/indexers/{colname}', indexers[colname], python_pickle_base='')

    if usetqdm:
        colnames = tqdm(coldata)
    else:
        colnames = coldata

    for colname in colnames:
        save(f'{folder}/columns/{colname}', coldata[colname], python_pickle_base='')


def coldata_load(folder, colnames=None, load_coldata=True, load_indexers=False, ignore_colnames=None, usetqdm=False):
    if usetqdm:
        from tqdm.notebook import tqdm
    import glob
    import os

    #     if ignore_colnames is None:
    #         ignore_colnames=[]

    if load_coldata:
        coldata = {}
        print(os.path.join(folder, 'columns/*.pickle'))
        if usetqdm:
            templist = tqdm(glob.glob(os.path.join(folder, 'columns/*.pickle')))
        else:
            templist = glob.glob(os.path.join(folder, 'columns/*.pickle'))

        for colfile in templist:
            colname = os.path.basename(colfile)[:-7]
            if ((colnames is None) or (colname in colnames)):
                coldata[colname] = load(colfile[:-7], python_pickle_base='')

        if not colnames is None:
            for colname in colnames:
                if not colname in coldata:
                    print(f'column not found: {colname}')
    if (load_indexers):
        if os.path.exists(os.path.join(folder, 'indexers.pickle')):  # Single file indexers
            print(os.path.join(folder, 'indexers.pickle'))
            indexers = load(os.path.join(folder, 'indexers'), python_pickle_base='')
        else:  # folder indexers
            indexers = {}
            print(os.path.join(folder, 'indexers/*.pickle'))
            for indexerfile in glob.glob(os.path.join(folder, 'indexers/*.pickle')):
                colname = os.path.basename(indexerfile)[:-7]
                if ((colnames is None) or (colname in colnames)):
                    indexers[colname] = load(indexerfile[:-7], python_pickle_base='')

    if load_coldata and load_indexers:
        return coldata, indexers
    elif load_coldata:
        return coldata
    elif load_indexers:
        return indexers


def convert_numpy_to_pure_python(obj):
    if (isinstance(obj, list)):
        return [convert_numpy_to_pure_python(elem) for elem in obj]
    if (isinstance(obj, tuple)):
        return tuple(convert_numpy_to_pure_python(elem) for elem in obj)
    if (isinstance(obj, np.ndarray)):
        return convert_numpy_to_pure_python(obj.tolist())
    if (isinstance(obj, dict)):
        res = {}
        for key, val in obj.items():
            res[key] = convert_numpy_to_pure_python(val)
        return res
    if isinstance(obj, np.generic):  # previous: if (isinstance(obj, np.number)):
        return obj.item()
    return obj


def mongo_encode(obj):  # make suitable to insert in mongo
    obj = convert_numpy_to_pure_python(obj)

    if (isinstance(obj, list)):
        return [mongo_encode(elem) for elem in obj]
    if (isinstance(obj, tuple)):
        return tuple(mongo_encode(elem) for elem in obj)
    if (isinstance(obj, dict)):
        res = {}
        for key, val in obj.items():
            key = str(key)
            if key.startswith('$') or key.startswith('.'):
                key = '\\' + key
            res[key] = mongo_encode(val)
        return res

    return obj


def mongo_decode(obj):  # remove backslash from start of keys
    obj = convert_numpy_to_pure_python(obj)

    if (isinstance(obj, list)):
        return [mongo_decode(elem) for elem in obj]
    if (isinstance(obj, dict)):
        res = {}
        for key, val in obj.items():
            if (key.startswith('\\')):
                key = key[1:]
            res[key] = mongo_decode(val)
        return res

    return obj


def flatten(d, parent_key='', sep='.', atom_check_func=None):
    """
    d: input dictionary.
    sep: the separator used between parent and child in making keys.
    atom_check_func: a function which checks an input and returns true if it should be considered as an atom and not be flattened.
    """

    if (atom_check_func == None):
        atom_check_func = lambda x: False

    #     print(inspect.getsourcelines(atom_check_func))
    #     print(atom_check_func(d),[isinstance(key, np.number) or isinstance(key, int) for key in d],
    #           all([isinstance(key, np.number) or isinstance(key, int) for key in d]), d)
    if atom_check_func(d):
        return d

    items = []
    if isinstance(d, list):
        d = {str(i): d[i] for i in range(len(d))}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if (isinstance(v, collections.MutableMapping) or isinstance(v, list)) and not atom_check_func(v):
            items.extend(flatten(v, new_key, sep=sep, atom_check_func=atom_check_func).items())
        else:
            items.append((new_key, v))
    return dict(items)


def is_sparse_matrix_json(d):
    return all([isinstance(key, np.number) or isinstance(key, int) for key in d])


def make_nonexisting_filename(file):  #
    import os
    import re
    cur_file = file
    while (os.path.exists(cur_file)):
        cur_name, extension = os.path.splitext(cur_file)
        m = re.match('(.*)(_)(\d+)', cur_name)
        original_name, _, number = m.groups()
        new_number = str(int(number) + 1)
        cur_file = original_name + '_' + new_number + extension
        cur_file

    return cur_file


"""
Hacky workaround to get the code of objects defined in interactive sessions of IPtyhon.
"""
import inspect, sys


def inspect_better_getfile(object, _old_getfile=inspect.getfile):
    if not inspect.isclass(object):
        return _old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, '__module__'):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, '__file__'):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + '.' + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    else:
        raise TypeError('Source for {!r} not found'.format(object))


# inspect.getfile = inspect_better_getfile


import inspect, sys


def inspect_better_getsource(object):
    import linecache  # Ipython uses linecache to save codes
    try:
        return inspect.getsource(object)
    except Exception as ex:
        file = inspect_better_getfile(object)
        if (file.startswith('<')):
            return ''.join(linecache.cache[file][2])  # Number 2 element of the returned tuple is the code
        else:
            raise (ex)


def jupyter_dbg_init_trace(globals_dict, suspend=False):
    """
    Move all the implicit functions to a specific file and start remote debugging of pycharm.
    """
    jupyter_move_implicit_codes_to_file(globals_dict);

    import pydevd_pycharm
    pydevd_pycharm.settrace('127.0.0.1', port=777, stdoutToServer=True, stderrToServer=True, patch_multiprocessing=True,
                            trace_only_current_thread=False, suspend=suspend)


def jupyter_move_implicit_codes_to_file(globals_dict):
    """
    The implicit classes and functions which were defined interactively can not be debugged in pycharm. To overcome this we move them to a real file.
    Important Note: The class definitions are kind of messy, thus they should be contained in their own cell!
    (or the whole cell will be executed again for each of the classes defined in it)
    """

    import inspect
    import os
    folder, jup_file = os.path.split(jupyter_find_file())
    module_name = 'temp_functions_' + jup_file
    module_name = module_name.replace('.', '_').replace(' ', '')
    module_file = f'{folder}/{module_name}.py'

    codes = []
    for name, obj in dict(globals_dict).items():
        file = ''

        try:
            #             print(obj.__name__)
            if (callable(obj) or inspect.isclass(obj)):
                file = inspect_better_getfile(obj)
                code = inspect_better_getsource(obj)
                #                 print(obj.__name__,file)
                if (file.startswith('<') or file.endswith(module_file)):
                    if (obj.__name__ != name):  #
                        #                     print(f'warning: function was not saved as just directly defined functions are supprted currently (obj.__name__ != name): {obj.__name__ } != {name} ')
                        continue
                    #                 print(name,file)

                    codes.append(code)
        #                     print('***',obj.__name__,code)

        except Exception as ex:
            pass

    codes = '\n'.join(codes)
    with open(module_file, 'w') as f:
        f.writelines(codes)

    #     exec(f"""
    # import {module_name}
    # import importlib
    # importlib.reload({module_name})
    # from {module_name} import *
    #     """, globals_dict)

    code = compile(codes, module_file, 'exec')
    exec(code, globals_dict)
    return module_file, codes


def class_fullname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


import types


def object_update_class_def(obj, cls):
    obj_methods = {}
    for name, method in inspect.getmembers(obj, predicate=inspect.ismethod):  # pay attention: is"method"
        obj_methods[name] = inspect.getsourcelines(method)

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):  # pay attention: is"function"
        source = inspect.getsourcelines(method)
        if (not name in obj_methods or obj_methods[name] != source):
            print(name)
            obj.__setattr__(name, types.MethodType(method, obj))


def freeze(d):  # freeze dict and list to make them hashable
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d


import types


# noglobal = lambda f: types.FunctionType(f.__code__, {}, argdefs=f.__defaults__)

def imports():
    for name, val in globals().items():
        # module imports
        if isinstance(val, types.ModuleType):
            yield name, val
        # functions / callables
        if hasattr(val, '__call__'):
            yield name, val


noglobal = lambda fn: types.FunctionType(fn.__code__, dict(imports()))


def np_nan(n):
    res = np.zeros(n)
    res.fill(np.nan)
    return res


def moving_average(x, N):
    #     return np.convolve(x, np.ones((N,))/N, mode='valid')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# # sample function
# @noglobal
# def f():
#     np.zeros
#     return x

# Source: https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
class defaultdict_keyed(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def remove_none_fields(dic):
    res = {}
    for key, val in dic.items():

        if (val is None):
            continue
        elif (isinstance(val, dict)):
            res[key] = remove_none_fields(val)
        elif (isinstance(val, list)):
            res[key] = []
            for i, elem in enumerate(val):
                if (isinstance(elem, dict)):
                    res[key].append(remove_none_fields(elem))
                else:
                    res[key].append(elem)
        else:
            res[key] = val
    return res
