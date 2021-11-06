"""
Running the mlservice which monitors the mongo collection and run the new jobs.
"""

from __future__ import print_function, division
from pymongo import MongoClient

from mlservice.common.util import *
import mlservice.ml_service_util
from mlservice.ml_service_util import create_job_for_new_collections

# R&D database
username_rd = urllib.parse.quote_plus('tapsellAI')
password_rd = urllib.parse.quote_plus('N3JeNLPZFaU5TsBM27RynHZgwUdmCdp6')
mongoAuthURL_rd = 'mongodb://127.0.0.1:27017'  # 'mongodb://%s:%s@127.0.0.1:32768'% (username_rd, password_rd)
client_rd = MongoClient(mongoAuthURL_rd)
db_features = client_rd.get_database('feature-engineering')
jobs_coll = db_features['_learning_jobs']

indexers = {}

while True:
    time.sleep(0.1)

    create_job_for_new_collections(db_features)
    for learning_job in jobs_coll.find():
        mlservice.ml_service_util.process_learning_job_doc(learning_job, jobs_coll, indexers)
