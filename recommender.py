import scipy.spatial.distance as sc
import datetime
import cloudpickle as pkl
import os

from utils import *
from sklearn.metrics import recall_score, precision_score


class Recommender(object):
    top_N = 10

    def __init__(self):
        self.users_dealitem_ids = {}
        self.dealitem_users_ids = {}
        self.top_n_items = []
        self.recall_scores = []
        self.precision_scores = []
        self.distance_matrix = []

    def fit(self, grouped_by_users, grouped_by_items, top_N_items=top_N):
        self.grouped_by_users_ids = grouped_by_users
        self.grouped_by_items_ids = grouped_by_items

        prt("Fitting model with %d users and %d items..." % (len(grouped_by_users), len(grouped_by_items)))

        # for user in self.grouped_by_users_ids:
        #     self.users_dealitem_ids.append({user[0]: list(user[1].groupby('dealitem_id').groups.keys())})
        #
        # for user in self.grouped_by_items_ids:
        #     self.dealitem_users_ids.append({user[0]: list(user[1].groupby('user_id').groups.keys())})

        for user in self.grouped_by_users_ids:
            self.users_dealitem_ids.update({str(user[0]): list(user[1].groupby('dealitem_id').groups.keys())})

        for item in self.grouped_by_items_ids:
            self.dealitem_users_ids.update({str(item[0]): list(item[1].groupby('user_id').groups.keys())})

        # sort deals
        # self.dealitem_users_ids.sort(key=lambda t: len(self.dealitem_users_ids[t.keys()[0]]), reverse=True)

        self.top_n_items = list(self.dealitem_users_ids)[:top_N_items]
        prt("Fitting model finished successfully.")

    def predict(self, x, y, top_N_items=top_N, distance_treshold=0.5, show_null_values=False):
        prt("Predicting...")

        distance_matrix_file = "distance_matrix.pickle"

        if os.path.exists(distance_matrix_file):
            with open(distance_matrix_file, "rb") as input_file:
                self.distance_matrix = pkl.load(input_file)

        if not self.distance_matrix:
            user_i = 0
            for user in y:
                user_i = user_i + 1
                recommended_items = []

                # new user. recommend top items
                if str(user[0]) not in self.users_dealitem_ids:
                    recommended_items = self.top_n_items
                else:
                    user_items = self.users_dealitem_ids[str(user[0])]

                    if len(user_items) < 2:
                        recommended_items = self.top_n_items
                    else:
                        N_items = len(user_items) * 2
                        recommended_items = self.recommed_items(user_items, user[0], treshold=distance_treshold)

                u = recommended_items
                v = list(user[1].groupby('dealitem_id').groups.keys())

                y_pred, y_true = get_vectors(u, v)

                recall = recall_score(y_true, y_pred, average='macro')
                precision = precision_score(y_true, y_pred, average='macro')

                self.recall_scores.append(recall)
                self.precision_scores.append(precision)

                hits = 0
                for rec_item in v:
                    if rec_item in u:
                        hits = hits + 1

                if show_null_values or (hits > 0 and recall != 0 and precision != 0):
                    prt("User [%d] with precision(%f), recall(%f) and hits(%d/%d) from %d users" % (user[0], precision, recall, hits, len(v), user_i))

            with open(distance_matrix_file, "wb") as output_file:
                pkl.dump(self.distance_matrix, output_file)

    def recommed_items(self, user_items, user, treshold=0.5):
        N_items = len(user_items) * 2
        similar_user_ids = []

        for item in user_items:
            similar_user_ids.append(self.dealitem_users_ids[str(item)])

        distances = []
        for i in [val for sublist in similar_user_ids for val in sublist]:
            if i != user:
                u, v = get_vectors(self.users_dealitem_ids[str(i)], user_items)
                distances.append([i, sc.cosine(u, v)])

        distances.sort(key=lambda t: t[1])
        self.distance_matrix.append({user: distances[:20]})

        recommended_items = []

        for i in distances:
            # top_df = x.loc[x['user_id'] == i[0]]
            # top_item = top_df['dealitem_id'].values[0]
            top_items = self.users_dealitem_ids[str(i[0])]

            for item in top_items:
                # not in recommended items.
                # do not recommend already bought item
                if item not in recommended_items and item not in user_items and i[0] >= treshold:
                    recommended_items.append(item)
                if (len(recommended_items) > N_items):
                    break

            if (len(recommended_items) > N_items):
                break

        return recommended_items
