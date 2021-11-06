folder='/root/data/cafe-video/training-data/recent_concatenated'

from pymongo import MongoClient
import urllib
from bson import ObjectId
from datetime import datetime
import dill
import gridfs

import mlservice.common.util as util
# from util import *


from mlservice.stages import sklearn_stages as mls
from mlservice.stages import cafevideo

coldata, indexers = util.coldata_load(folder, load_indexers=True, usetqdm=False)

pipeline=cafevideo.get_cafe_video_conversion_estimation_pipeline(coldata)
pipeline.fit_transform(coldata)

# ------------- Save the model ------------------
username, password = ('admin', 'cLVabiJ1IiqJYHldbfhduvadhj40UyXcLVabiJ1I') #('admin', 'rALU7VYXnyq9ugQ0ItYUIDYzEr4')
server='172.16.19.201'
port = 27017  # 27019
database = 'mlaas'

# username = urllib.parse.quote_plus('username')
# password = urllib.parse.quote_plus('pass')
mongoAuthURL = f'mongodb://{username}:{password}@{server}:{port}'

client = MongoClient(mongoAuthURL)
db = client.get_database(database)

mongo_coll_pipeline_def = db['pipeline_definitions']
gfs = gridfs.GridFS(db)
gridfsObjectId = gfs.put(dill.dumps(pipeline))

doc = {
    'tags': ['CafeVideo'],
    'modifiedTime': datetime.utcnow(),
    'definition': {
        'stages': [
            {'GridFsLink': {
                'pickleFileId': gridfsObjectId}
            }
        ]
    }
}

pipelineId = mongo_coll_pipeline_def.insert_one(doc).inserted_id

print(pipelineId)
