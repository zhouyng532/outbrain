import pandas as pd
import pymc3 as pm
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# prior: the gamma distribution of probability of being clicked for a specific advertiser (mean and std)
clicks_train = pd.read_csv('raw_data/clicks_train.csv')

ad_click_cnt = clicks_train[clicks_train['clicked'] == 1].ad_id.value_counts()
ad_click_cntall = clicks_train.ad_id.value_counts()

def get_prob(ad_id):
    if ad_id not in ad_click_cnt:
        return 0
    return ad_click_cnt[ad_id]/float(ad_click_cntall[ad_id])

promoted_content = pd.read_csv('raw_data/promoted_content.csv', usecols=['ad_id','advertiser_id'])

cnt = []
for a in promoted_content['ad_id']:
    cnt.append(get_prob(a))
promoted_content['cnt']=pd.Series(cnt,index = promoted_content.index)

advertiser_prior_mean = promoted_content.groupby(['advertiser_id'])['cnt'].mean()
advertiser_prior_std = promoted_content.groupby(['advertiser_id'])['cnt'].std()

max(advertiser_prior_mean) # 1.0
max(advertiser_prior_std) # 0.70710678118654757
min(advertiser_prior_mean) # 0.0
min(advertiser_prior_std) # 0.0

advertiser_prior_mean.mean() # 0.12644614288567343
advertiser_prior_mean.std() # 0.11647595665698497
advertiser_prior_std.mean() # 0.1270377887303322
advertiser_prior_std.std() # 0.10042517820584072

# ---------------------------------------------------------------------------------------------------------------------
# data
ad_data = clicks_train.groupby(['ad_id']).agg({'clicked': (lambda x: list(x))})

# dictionary to connect ad_id and advertiser_id
ad_avt = promoted_content.copy()
ad_avt.index = ad_avt['ad_id']
ad_avt = ad_avt[['advertiser_id']]
ad_avt = ad_avt.to_dict()['advertiser_id']


# ---------------------------------------------------------------------------------------------------------------------
a = 11
# bayesian model
model = None
with pm.Model()as model:
    mu = pm.Normal('mean', mu=advertiser_prior_mean[ad_avt[a]], sd=advertiser_prior_mean.std()*2)
    sigma = pm.Gamma('std', mu=advertiser_prior_std[ad_avt[a]], sd=advertiser_prior_std.std())
    likelihood = pm.Gamma('click_prob', mu=mu, sd=sigma,
                           observed=np.array(ad_data.ix[a,'clicked']))
    trace = pm.sample(5000, step=pm.Metropolis(), progressbar=False, random_seed=16357)

trace.get_values('mean').mean()



