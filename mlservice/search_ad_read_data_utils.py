"""
Utils specific for searchad
"""
import urllib
from pymongo import MongoClient
from mlservice.common import util
from datetime import datetime

username = urllib.parse.quote_plus('admin')
password = urllib.parse.quote_plus('S3rcHM0ng0AdDb')
mongoAuthURL = 'mongodb://%s:%s@172.24.111.41' % (username, password)
mongoURL = 'mongodb://172.24.111.41'

client = MongoClient(mongoAuthURL)
db = client.get_database('search-ad')

appCategories = [
    'BOOKS_AND_REFERENCE',
    'BUSINESS',
    'COMMUNICATION',
    'EDUCATION',
    'ENTERTAINMENT',
    'FINANCE',
    'FOOD_AND_DRINK',
    'GAME_ACTION',
    'GAME_ADVENTURE',
    'GAME_ARCADE',
    'GAME_BOARD',
    'GAME_CARD',
    'GAME_CASUAL',
    'GAME_EDUCATIONAL',
    'GAME_FAMILY',
    'GAME_PUZZLE',
    'GAME_RACING',
    'GAME_SIMULATION',
    'GAME_SPORTS',
    'GAME_STRATEGY',
    'GAME_WORD',
    'HEALTH_AND_FITNESS',
    'LIFESTYLE',
    'MAPS_AND_NAVIGATION',
    'MEDICAL',
    'MUSIC_AND_AUDIO',
    'NEWS_AND_MAGAZINES',
    'PERSONALIZATION',
    'PHOTOGRAPHY',
    'PRODUCTIVITY',
    'RELIGIOUS',
    'SHOPPING',
    'SOCIAL',
    'SPORTS',
    'TOOLS',
    'TRAVEL_AND_LOCAL',
    'UNKNOWN',
    'VIDEO_PLAYERS',
    'WEATHER']

cat2unifiedCat = {'کتابها و منابع': 'BOOKS_AND_REFERENCE',
                  'کلمات و دانستنیها': 'GAME_WORD',
                  'پیامرسانها': 'COMMUNICATION',
                  'شخصیسازی': 'PERSONALIZATION',
                  'شبکههای اجتماعی': 'SOCIAL',
                  'شبیهسازی': 'GAME_SIMULATION',
                  'productivity': 'PRODUCTIVITY',
                  'entertainment': 'ENTERTAINMENT',
                  'medical': 'MEDICAL',
                  'personalization': 'PERSONALIZATION',
                  'religious': 'RELIGIOUS',
                  'travel-local': 'TRAVEL_AND_LOCAL',
                  'casual': 'GAME_CASUAL',
                  'family': 'GAME_FAMILY',
                  'maps-navigation': 'MAPS_AND_NAVIGATION',
                  'news': 'NEWS_AND_MAGAZINES',
                  'action': 'GAME_ACTION',
                  'shopping': 'SHOPPING',
                  'arcade': 'GAME_ARCADE',
                  'puzzle': 'GAME_PUZZLE',
                  'media-video': 'VIDEO_PLAYERS',
                  'music-audio': 'MUSIC_AND_AUDIO',
                  'racing': 'GAME_RACING',
                  'simulation': 'GAME_SIMULATION',
                  'sports-game': 'GAME_SPORTS',
                  'business': 'BUSINESS',
                  'health-fitness': 'HEALTH_AND_FITNESS',
                  'adventure': 'GAME_ADVENTURE',
                  'books-reference': 'BOOKS_AND_REFERENCE',
                  'photography': 'PHOTOGRAPHY',
                  'sports': 'SPORTS',
                  'word-trivia': 'GAME_WORD',
                  'strategy': 'GAME_STRATEGY',
                  'social': 'SOCIAL',
                  'lifestyle': 'LIFESTYLE',
                  'communication': 'COMMUNICATION',
                  'education': 'EDUCATION',
                  'tools': 'TOOLS',
                  'educational': 'GAME_EDUCATIONAL',
                  'finance': 'FINANCE',
                  'food-drink': 'FOOD_AND_DRINK',
                  'weather': 'WEATHER'}

assert (set(list(cat2unifiedCat.values())).issubset(set(appCategories)))

# categoriesToInd={cat:ind for cat,ind in zip(appCategories,range(len(appCategories)))}
# categoriesToInd.update({badCat:categoriesToInd[correctCat] for badCat,correctCat in cat2unifiedCat.items()})


# ind2category=[None]*len(categoriesToInd)
# for cat,ind in categoriesToInd.items():
#     ind2category[ind]=cat

import numpy as np

col2dtype = {
    'requestId': np.uint32,
    'creationDate': np.uint64,
    'adCategory': np.uint8,
    'adRatingInStore': np.float32,
    'adTapRate': np.float64,
    'isQueryInGarbageGroup': np.bool,
    'adInstallsInStore': np.uint,
    'adInstallRate': np.float64,
    'hourOfDay': np.uint8,
    'isQueryAppName': np.bool,
    'dayOfWeek': np.uint8,
    'queryAdSimBasedOnCategory': np.bool,
    'holiday': np.bool,
    'queryAdTapRate': np.float64,
    'tapped': np.bool,
    'adPackageName': np.uint32,
    'query': np.uint32,
    'matchedQueryId': np.uint32
}


def read_coldata_request_ad_feature(indexers, limit_docs=None):
    from tqdm import tqdm_notebook as tqdm

    elem = db['request_ad_feature'].find_one()
    colnames = list(col2dtype.keys())  # list(set(sdf.columns).intersection(set(list(elem.keys()))))+['tapped']

    n = int(db['request_ad_feature'].estimated_document_count() * 1.1)
    print("db['request_ad_feature'].estimated_document_count() => ",
          db['request_ad_feature'].estimated_document_count())

    coldata = {}
    for colname in colnames:
        coldata[colname] = np.zeros(n, dtype=col2dtype[colname])

    if (limit_docs is None):
        stream = db['request_ad_feature'].find()
    else:
        stream = db['request_ad_feature'].find().limit(limit_docs)
    for i, elem in tqdm(enumerate(stream)):
        for colname in colnames:
            #         if(colname in ['adCategory']):
            #             continue
            val = elem[colname]
            if (colname == 'adCategory'):
                val = cat2unifiedCat.get(val, val)
            if (isinstance(val, datetime)):
                val = int(util.datetime2milis(val))

            if (colname in indexers):
                val = indexers[colname].add(val)

            coldata[colname][i] = val

    for colname in colnames:
        coldata[colname] = coldata[colname][:i + 1]

    coldata['matchedQueryId'] += 1

    return coldata
