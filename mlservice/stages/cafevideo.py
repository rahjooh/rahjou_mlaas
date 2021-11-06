from mlservice.stages import sklearn_stages as mls
import scipy as sp
import numpy as np


def get_dim(coldata, sparse_colname, dim_colname):
    if not coldata[dim_colname].flags.writeable:
        coldata[dim_colname] = coldata[dim_colname].copy()  # some wierd error will happen otherwise

    clipped = np.clip(coldata[dim_colname], 0, coldata[sparse_colname].shape[1] - 1)
    n = coldata[dim_colname].shape[0]  # number of samples
    tmp = coldata[sparse_colname][np.arange(n), clipped]
    tmp = tmp.todense() if isinstance(tmp, sp.sparse.base.spmatrix) else tmp
    res = np.asarray(tmp)[0]
    res[coldata[dim_colname] > coldata[sparse_colname].shape[1] - 1] = 0
    return res


class CafeVideoExtraFeatures(mls.Stage):

    def __init__(self, coldata):
        super().__init__(
            input=[colname for colname in coldata if
                   'adGenreFeatures' in colname and colname.endswith('impressions.count')] \
                  + [colname for colname in coldata if
                     'adGenreFeatures' in colname and colname.endswith('clicks.count')] \
                  + [colname for colname in coldata if
                     'adGenreFeatures' in colname and colname.endswith('impressions.recentCount')] \
                  + [colname for colname in coldata if
                     'adGenreFeatures' in colname and colname.endswith('clicks.recentCount')] \
                  + ['appGenre', 'userProfile.appHistoryInstallFeature.appGenreFeatures.installations',
                     'userProfile.appHistoryActivityFeature.appGenreFeatures.appActivity.totalDurationRecent']
            ,
            output=[
                'user_adGenre_impressions',
                'user_adGenre_impressions_recent',
                'user_adGenre_ctr',
                'user_adGenre_ctr_recent',
                'user_adGenre_app_installations',
                'user_adGenre_app_activity_recent_duration'
            ])

    def transform(self, coldata):
        ad_genre_impressions = (sum(get_dim(coldata, colname, 'appGenre') for colname in coldata if
                                    'adGenreFeatures' in colname and colname.endswith('impressions.count')) > 0)
        ad_genre_clicks = (sum(get_dim(coldata, colname, 'appGenre') for colname in coldata if
                               'adGenreFeatures' in colname and colname.endswith('clicks.count')) > 0)
        coldata['user_adGenre_impressions'] = ad_genre_impressions
        sw = ad_genre_impressions != 0
        coldata['user_adGenre_ctr'] = np.zeros_like(ad_genre_impressions, dtype=np.float)
        coldata['user_adGenre_ctr'][sw] = ad_genre_clicks[sw] / ad_genre_impressions[sw]
        coldata['user_adGenre_ctr'][~sw] = np.nan

        ad_genre_impressions = (sum(get_dim(coldata, colname, 'appGenre') for colname in coldata if
                                    'adGenreFeatures' in colname and colname.endswith('impressions.recentCount')) > 0)
        ad_genre_clicks = (sum(get_dim(coldata, colname, 'appGenre') for colname in coldata if
                               'adGenreFeatures' in colname and colname.endswith('clicks.recentCount')) > 0)
        coldata['user_adGenre_impressions_recent'] = ad_genre_impressions
        sw = ad_genre_impressions != 0
        coldata['user_adGenre_ctr_recent'] = np.zeros_like(ad_genre_impressions, dtype=np.float)
        coldata['user_adGenre_ctr_recent'][sw] = ad_genre_clicks[sw] / ad_genre_impressions[sw]
        coldata['user_adGenre_ctr_recent'][~sw] = np.nan

        coldata['user_adGenre_app_installations'] = get_dim(coldata,
                                                            'userProfile.appHistoryInstallFeature.appGenreFeatures.installations',
                                                            'appGenre')
        coldata['user_adGenre_app_activity_recent_duration'] = get_dim(coldata,
                                                                       'userProfile.appHistoryActivityFeature.appGenreFeatures.appActivity.totalDurationRecent',
                                                                       'appGenre')


def get_cafe_video_conversion_estimation_pipeline(coldata):
    stages = []
    features = []

    extendFeatures = CafeVideoExtraFeatures(coldata)
    stages.append(extendFeatures)

    for colname in ['user_adGenre_ctr', 'user_adGenre_ctr_recent']:
        stages.append(mls.Impute(input=colname, output=colname))

    for colname in extendFeatures.output:
        stages.append(mls.Quantize(input=colname, output=colname + '_q', label='events.click', len=10))
        stages.append(mls.Onehot(input=colname + '_q', output=colname + '_1hot'))
        features.append(colname + '_1hot')

    for colname in ['creativeIdIndex', 'context.zoneId']:
        stages.append(mls.Onehot(input=colname, output=colname + '_1hot'))
        features.append(colname + '_1hot')

    stages.append(mls.Learner(input=features, label='events.click', output='estimatedCVR.impressionToClick',
                              model='sklearn.linear_model.LogisticRegression', params={'max_iter': 1000}))

    return mls.Pipeline(stages=stages)
