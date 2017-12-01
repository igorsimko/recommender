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
actual_time = activity_train['create_time'].max()
# acctual_time = 1406852020
model = r.Recommender(actual_time)
model.fit(full_data, grouped_by_users_train, grouped_by_dealitem_id_train, deal_items_train, deal_details_train, top_N_items=N_dealitems)
model.predict(activity_train, grouped_by_users_test, distance_treshold=0.4)
