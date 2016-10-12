import pandas as pd

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
# 0: 224814; 1: 89254; 2: 39634


# distribution of clicks -- campaign_id
train_ad_campaign = pd.DataFrame(train_ad_info.groupby('campaign_id')['clicked'].sum())
train_ad_campaign.hist('clicked', bins = 100)


# distribution of clicks -- advertiser_id
train_ad_advertiser = pd.DataFrame(train_ad_info.groupby('advertiser_id')['clicked'].sum())
train_ad_advertiser.hist('clicked', bins = 100)

