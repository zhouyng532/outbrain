import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import pymc3 as pm


# 2016.10.12
# ---------------------------------------------------------------------------------------------------------------------
# calculate number of clicks for every ad_id appeared in train set
clicks_train = pd.read_csv('raw_data/clicks_train.csv', usecols = ['ad_id','clicked'])
ad_clicked = pd.DataFrame(clicks_train.groupby(by = 'ad_id')['clicked'].sum())
ad_clicked['ad_id'] = ad_clicked.index
# len(ad_clicked) = 478950

# ad_id specific characteristics
promoted_content = pd.read_csv('raw_data/promoted_content.csv', usecols = ['ad_id','campaign_id','advertiser_id'])
# len(promoted_content) = 559583

# merge two tables
train_ad_info = pd.merge(ad_clicked, promoted_content, how = 'left', on = 'ad_id')

# ---------------------------------------------------------------------------------------------------------------------

# distribution of clicks -- ad_id
train_ad_info.hist('clicked', bins = 100)
# 0: 224814; 1: 89254; 2: 39634 ...


# distribution of clicks -- campaign_id
train_ad_campaign = pd.DataFrame(train_ad_info.groupby('campaign_id')['clicked'].sum())
train_ad_campaign.hist('clicked', bins = 100)


# distribution of clicks -- advertiser_id
train_ad_advertiser = pd.DataFrame(train_ad_info.groupby('advertiser_id')['clicked'].sum())
train_ad_advertiser.hist('clicked', bins = 100)

# 2016.10.17
# ---------------------------------------------------------------------------------------------------------------------
# cluster ads only with advertiser_id (promoted_content, 4385 unique)
promoted_content = pd.read_csv('raw_data/promoted_content.csv')
clicks_train = pd.read_csv('raw_data/clicks_train.csv')

clicks_train = clicks_train[clicks_train['clicked'] == 1]

merged = pd.merge(promoted_content, clicks_train, how = 'left', on = 'ad_id').fillna(0)
merged = merged.groupby(['advertiser_id'])['clicked'].sum()

merged.hist(bins = 100) # number of clicks for each advertiser id
pass # number of ads for each advertiser id

page_views_sample = pd.read_csv('raw_data/page_views_sample.csv', usecols = ['uuid','document_id']) # unique
events = pd.read_csv('raw_data/events.csv', usecols = ['display_id','uuid','document_id']) # unique

# 2016.10.20
# ---------------------------------------------------------------------------------------------------------------------
# in train and test set, whats the proportion of ads that has campaign/advertiser info?
clicks_train = pd.read_csv('raw_data/clicks_train.csv', usecols = ['ad_id']).drop_duplicates() # 478950 unique id
clicks_test = pd.read_csv('raw_data/clicks_test.csv', usecols = ['ad_id']).drop_duplicates() # 381385 unique id

promoted_content = pd.read_csv('raw_data/promoted_content.csv', usecols = ['ad_id']) # 559583 unique id

len(pd.merge(clicks_train, promoted_content)) # all
len(pd.merge(clicks_test, promoted_content)) # all
len(pd.merge(clicks_test, clicks_train)) # 316035

# whats the proportion of ads that have ever linked to a document? Whats the distribution of number of links?
    # train set
clicks_train = pd.read_csv('raw_data/clicks_train.csv', usecols = ['ad_id','display_id'])
    # link ad and display
train_id = clicks_train.groupby('ad_id', as_index=False)['display_id'].agg({'display_id':(lambda x: list(x)),
                                                                            'num_display':(lambda x: len(x))})
    # link display and document
with open('raw_data/events.csv', mode = 'r') as infile: # read in as dict
    reader = csv.reader(infile)
    next(reader, None) # skip the header
    display_docs = {}
    for row in reader:
        display_docs[int(row[0])] = int(row[2])
document_id = []
num_doc = []
for d in train_id['display_id']:
    docs = [display_docs[x] for x in d]
    num_doc.append(len(set(docs)))
    document_id.append(docs)
train_id['document_id'] = pd.Series(document_id, index = train_id.index)
train_id['num_doc'] = pd.Series(num_doc, index = train_id.index)
train_id.index = train_id['ad_id']

    # test set
clicks_test = pd.read_csv('raw_data/clicks_test.csv', usecols = ['ad_id','display_id'])
    # link ad and display
test_id = clicks_test.groupby('ad_id', as_index=False)['display_id'].agg({'display_id':(lambda x: list(x)),
                                                                            'num_display':(lambda x: len(x))})
    # link display and document
document_id = []
num_doc = []
for d in test_id['display_id']:
    docs = [display_docs[x] for x in d]
    num_doc.append(len(set(docs)))
    document_id.append(docs)
test_id['document_id'] = pd.Series(document_id, index = test_id.index)
test_id['num_doc'] = pd.Series(num_doc, index = test_id.index)
test_id.index = test_id['ad_id']

unique_ad_id = pd.read_csv('raw_data/promoted_content.csv', usecols = ['ad_id'])['ad_id'].tolist()

train_test_merged = pd.merge(train_id, test_id, how='outer',on='ad_id', suffixes=('_train','_test'))

# display in train and test
_display = train_test_merged[['num_display_train','num_display_test']].fillna(0)
_display.index = train_test_merged['ad_id']
plt.plot(_display.index, _display['num_display_train'], 'g-',label = 'train')
plt.plot(_display.index, _display['num_display_test'], 'b-',label = 'test')
plt.legend()
plt.show()

# docs in train and test
_docs = train_test_merged[['num_doc_train','num_doc_test']].fillna(0)
_docs.index = train_test_merged['ad_id']
fig, ax = plt.subplots(1,1)
ax.plot(_docs.index, _docs['num_doc_train'], ls = '-', label = 'train', color = 'g')
ax.plot(_docs.index, _docs['num_doc_test'], ls = '-', label = 'test', color = 'b')
ax.legend()
fig.show()


# 2016.11.12
# ---------------------------------------------------------------------------------------------------------------------
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

# dictionary to connect ad and cluster
ad_clust = promoted_content.copy()
ad_clust.index = ad_clust['ad_id']
ad_clust = ad_clust[['cluster']]
ad_clust = ad_clust.to_dict()['cluster']

# data
ad_data = clicks_train.groupby(['ad_id']).agg({'clicked': (lambda x: list(x))})
ad = 2
ad_cluster = ad_clust[ad]

model = None
with pm.Model()as model:
    mu = pm.Normal('mean', mu=cluster_df.ix[ad_cluster,:]['prob_mean_scale'], sd=cluster_df['prob_mean_scale'].std()/4)
    sigma = pm.Gamma('std', mu=cluster_df.ix[ad_cluster,:]['prob_std'], sd=cluster_df['prob_std'].std())
    likelihood = pm.Gamma('click_prob', mu=mu, sd=sigma,
                           observed=np.array(ad_data.ix[ad,'clicked']))
    trace = pm.sample(5000, step=pm.Metropolis(), progressbar=False, random_seed=16357)

trace.get_values('mean').mean() # check the mean value of sampling
