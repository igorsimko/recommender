import pandas as pd
import matplotlib.pyplot as plt
import processing
import recommender as r
from itertools import groupby
import numpy as np
from utils import *

import datetime

from sklearn.metrics import recall_score, precision_score
import scipy.spatial.distance as sc

# - natrenovat iba na traine ignorovat test pre train
# - chceme predicovat deal nie tu najnizsiu uroven
# - chellenge - trenovanie na train+test

# vysledky v jednotkach percentach su dobre, radovo v desiatkach uz bude nieco zle

__ACTIVITY = "activity_v2.csv"
__DEAL_ITEMS = "dealitems.csv"
__DEAL_DETAILS = "deal_details.csv"

# Params
N_dealitems = 10

# load raw data
activity_train = pd.read_csv('train_' + __ACTIVITY)
deal_items_train = pd.read_csv('train_' + __DEAL_ITEMS)
deal_details_train = pd.read_csv('train_' + __DEAL_DETAILS)

activity_test = pd.read_csv('test_' + __ACTIVITY)
deal_items_test = pd.read_csv('test_' + __DEAL_ITEMS)
deal_details_test = pd.read_csv('test_' + __DEAL_DETAILS)

full_data, grouped_by_users_train, grouped_by_dealitem_id_train = processing.get_proceed_data(activity_train,
                                                                                              deal_items_train,
                                                                                              deal_details_train)
_, grouped_by_users_test, grouped_by_dealitem_id_test = processing.get_proceed_data(activity_test,
                                                                                                 deal_items_test,
                                                                                                 deal_details_test)

model = r.Recommender()
model.fit(grouped_by_users_train, grouped_by_dealitem_id_train)
model.predict(full_data, grouped_by_users_test)

# #### TRAIN #####
# users_dealitem_ids = []
# dealitem_users_ids = []
#
# for user in grouped_by_users_train:
#     users_dealitem_ids.append({user[0]: list(user[1].groupby('dealitem_id').groups.keys())})
#     # list(user[1].groupby('dealitem_id').groups.keys())
#
# dealitem_users_ids = []
#
# for item in grouped_by_dealitem_id_train:
#     dealitem_users_ids.append({item[0]: list(item[1].groupby('user_id').groups.keys())})
#     # list(user[1].groupby('dealitem_id').groups.keys())
#
# # sort deals
# # dealitem_users_ids.sort(key=lambda t: len(dealitem_users_ids[t]), reverse=True)
#
# top_n_dealitems = dealitem_users_ids[:N_dealitems]
#
# users_dealitem_ids_test = []
#
# ### TEST ###
# recall_scores = []
# precision_scores = []
#
# # groups_users = grouped_by_users_train.groups
# # groups_items = grouped_by_dealitem_id_train.groups
#
# for user in grouped_by_users_test:
#     recommended_items = []
#     user_items = list(user[1].groupby('dealitem_id').groups.keys())
#
#     distances = []
#     for i in users_dealitem_ids:
#         if list(i.keys())[0] != user[0]:
#             u, v = get_vectors(list(i.values())[0], user_items)
#             distances.append([list(i.keys())[0], sc.cosine(u, v)])
#
#     distances.sort(key=lambda t: t[1])
#
#     for i in distances:
#         top_df = full_data.loc[full_data['user_id'] == i[0]]
#         top_item = top_df['dealitem_id'].values[0]
#
#         # not in recommended items.
#         # do not recommend already bought item
#         if top_item not in recommended_items and top_item not in user_items:
#             recommended_items.append(top_item)
#         # top_df['coupon_end_time']
#         # top_df['coupon_begin_time']
#
#     # users_dealitem_ids_test.append([user[0], list(user[1].groupby('dealitem_id').groups.keys())])
#
#
#     y_true = 0 # list(recommended_items)
#     y_pred = 1 # list(user[1].groupby('dealitem_id').groups.keys())
#
#     recall_scores.append(recall_score(y_true, y_pred, average='macro'))
#     precision_scores.append(precision_score(y_true, y_pred, average='macro'))

activity_test = []
deal_items_test = []
deal_details_test = []
