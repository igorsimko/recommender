import scipy.spatial.distance as sc
import datetime

from utils import *
from sklearn.metrics import recall_score, precision_score

class Recommender(object):
    top_N = 10

    def __init__(self):
        self.users_dealitem_ids = []
        self.dealitem_users_ids = []
        self.top_n_items = []
        self.recall_scores = []
        self.precision_scores = []

    def fit(self, grouped_by_users, grouped_by_items, top_N_items=top_N):
        self.grouped_by_users_ids = grouped_by_users
        self.grouped_by_items_ids = grouped_by_items
        prt("Fitting model with %d users and %d items..." % (len(grouped_by_users), len(grouped_by_items) ))

        # for user in self.grouped_by_users_ids:
        #     self.users_dealitem_ids.append({user[0]: list(user[1].groupby('dealitem_id').groups.keys())})
        #
        # for user in self.grouped_by_items_ids:
        #     self.dealitem_users_ids.append({user[0]: list(user[1].groupby('user_id').groups.keys())})

        for user in self.grouped_by_users_ids:
            self.users_dealitem_ids.append({user[0]: list(user[1].groupby('dealitem_id').groups.keys())})

        for item in self.grouped_by_items_ids:
            self.dealitem_users_ids.append({item[0]: list(item[1].groupby('user_id').groups.keys())})

        # sort deals
        # self.dealitem_users_ids.sort(key=lambda t: len(self.dealitem_users_ids[t.keys()[0]]), reverse=True)
        self.top_n_items = self.dealitem_users_ids[:top_N_items]
        prt("Fitting model finished successfully.")

    def predict(self, x, y, top_N_items=top_N):
        prt("Predicting...")

        self.top_n_items = self.dealitem_users_ids[:top_N_items]

        for user in y:
            recommended_items = []
            user_items = list(user[1].groupby('dealitem_id').groups.keys())

            distances = []
            for i in self.users_dealitem_ids:
                if list(i.keys())[0] != user[0]:
                    u, v = get_vectors(list(i.values())[0], user_items)
                    distances.append([list(i.keys())[0], sc.cosine(u, v)])

            distances.sort(key=lambda t: t[1])

            for i in distances:
                top_df = x.loc[x['user_id'] == i[0]]
                top_item = top_df['dealitem_id'].values[0]

                # not in recommended items.
                # do not recommend already bought item
                if top_item not in recommended_items and top_item not in user_items:
                    recommended_items.append(top_item)
                    # top_df['coupon_end_time']
                    # top_df['coupon_begin_time']
            prt("User [%d] with recommended items [" + " ".join([str(x) for x in recommended_items]) + "]")

            y_true = 0  # list(recommended_items)
            y_pred = 1  # list(user[1].groupby('dealitem_id').groups.keys())

            self.recall_scores.append(recall_score(y_true, y_pred, average='macro'))
            self.precision_scores.append(precision_score(y_true, y_pred, average='macro'))
