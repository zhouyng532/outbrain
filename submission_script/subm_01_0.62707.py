import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import pymc3 as pm

# cluster based on campaign
clicks_train = pd.read_csv('raw_data/clicks_train.csv')

ad_click_cnt = clicks_train[clicks_train['clicked'] == 1].ad_id.value_counts()
ad_click_cntall = clicks_train.ad_id.value_counts()

def get_prob(ad_id):
    if ad_id not in ad_click_cnt:
        return 0
    return ad_click_cnt[ad_id]/(float(ad_click_cntall[ad_id]) + 10)

promoted_content = pd.read_csv('raw_data/promoted_content.csv', usecols=['ad_id','campaign_id','advertiser_id'])

ad_prob = []
for a in promoted_content['ad_id']:
    ad_prob.append(get_prob(a))
promoted_content['ad_prob'] = pd.Series(ad_prob, index=promoted_content.index)

campaign_size = promoted_content.groupby(['campaign_id']).agg({'ad_id':(lambda x: len(x)),
                                                               'advertiser_id':(lambda x: list(x)[0]),
                                                               'ad_prob':(lambda x: x.mean())})

# start to do clustering
def FindClosestCam(campaign_id, ind_adt_lg, ind_adt_sg):
    ind_adt_lg['diff'] = abs(ind_adt_lg['ad_prob'] - ind_adt_sg.ix[campaign_id, 'ad_prob'])
    lg = ind_adt_lg[ind_adt_lg['diff'] == min(ind_adt_lg['diff'])].index[0]
    return lg


MIN = 50 # minimum size of original campaign group that can stand alone
campaign_cluster = dict() # dictionary that map campaign id and cluster id
i = 0 # cluster id

for adt in list(set(campaign_size['advertiser_id'])):
    ind_adt = campaign_size[campaign_size['advertiser_id'] == adt].sort_values('ad_id')
    if len(ind_adt) == 1:
        campaign_cluster[ind_adt.index[0]] = i
        i += 1
    elif ind_adt['ad_id'].tolist()[-2] <= MIN:
        for c in ind_adt.index:
            campaign_cluster[c] = i
        i += 1
    else:
        ind_adt_lg = ind_adt[ind_adt['ad_id'] > MIN]
        for c in ind_adt_lg.index:
            campaign_cluster[c] = i
            i += 1
        ind_adt_sg = ind_adt[ind_adt['ad_id'] <= MIN]
        for c in ind_adt_sg.index:
            lg = FindClosestCam(c, ind_adt_lg, ind_adt_sg)
            campaign_cluster[c] = campaign_cluster[lg]

# when MIN = 50, number of clusters = 5898
# when MIN = 60, number of cluster = 5639
# when MIN = 100, number of cluster = 4803

promoted_content['cluster'] = promoted_content['campaign_id'].map(lambda x: campaign_cluster[x])

cluster_df = promoted_content.groupby(['cluster']).agg({'ad_id': (lambda x: len(x)),
                                                        'campaign_id':(lambda x: list(set(x))),
                                                        'advertiser_id': (lambda x: list(x)[0]),
                                                        'ad_prob': [np.mean, np.std]})
cluster_df.columns = ['advertiser_id','prob_mean','prob_std','num_ad','campaign_id']
cluster_df['prob_mean_scale'] = preprocessing.minmax_scale(cluster_df['prob_mean'])


# only add up the cluster-average and individual ad probability
promoted_content['score'] = promoted_content['cluster'].map(lambda x: cluster_df.ix[x,'prob_mean_scale'])
promoted_content['final'] = promoted_content['ad_prob'] + promoted_content['score']

clicks_test = pd.read_csv('raw_data/clicks_test.csv')
ad_score = promoted_content.copy()
ad_score.index = ad_score['ad_id']
ad_score = ad_score[['final']]

def get_score(k):
    if k not in ad_score.index:
        return 0
    return ad_score.ix[k, 'final']

def SortByScore(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key = get_score, reverse = True)
    return " ".join(map(str, ad_ids))

ss = pd.read_csv('raw_data/sample_submission.csv')
ss['ad_id'] = ss.ad_id.apply(lambda x: SortByScore(x))
ss.to_csv('output_data/first_submission.csv', index = False)