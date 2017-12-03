import scipy.spatial.distance as sc
import datetime
import cloudpickle as pkl
import os
import operator

from utils import *
from sklearn.metrics import recall_score, precision_score, mean_squared_error

top_N = 10
MONTH = 2678400


class Recommender(object):
    def __init__(self, actual_time=1):
        self.users_dealitem_ids = {}
        self.dealitem_users_ids = {}
        self.coupons_in_time = {}
        self.ndcg_scores = {}
        self.deal_items = []
        self.full_data = []

        self.top_n_items = []
        self.deal_items = []
        # metrics
        self.precision_scores_default = []
        self.precision_scores_top = []
        self.precision_scores_top_w = []

        self.avg_hits_top_n = []
        self.avg_hits_top_w = []
        self.avg_hits_default = []

        self.distance_matrix = []
        self.actual_time = actual_time

    def fit(self, full_data, grouped_by_users, grouped_by_items, deal_items, deal_details, top_N_items=top_N):
        metadata = {}
        file_path = 'metadata.pkl'

        if os.path.exists(file_path):
            with open(file_path, "rb") as input_file:
                metadata = pkl.load(input_file)
        self.full_data = full_data
        self.grouped_by_users_ids = grouped_by_users
        self.grouped_by_items_ids = grouped_by_items
        self.deal_items = deal_items

        self.grouped_by_cities = deal_details.groupby(deal_details['title_city'].str.lower()).groups

        if not metadata:
            prt("Fitting model with %d users and %d items..." % (len(grouped_by_users), len(grouped_by_items)))

            for user in self.grouped_by_users_ids:
                self.users_dealitem_ids.update({str(user[0]): list(user[1].groupby('dealitem_id').groups.keys())})

            for item in self.grouped_by_items_ids:
                self.dealitem_users_ids.update({str(item[0]): list(item[1].groupby('user_id').groups.keys())})

            self.top_n_items, self.coupons_in_time = get_top_n_and_coupons(full_data,deal_items,self.actual_time,top_N_items)

            metadata = {
                'gusers': self.users_dealitem_ids,
                'gitems': self.dealitem_users_ids,
                'top': self.top_n_items,
                'coupons': self.coupons_in_time
            }
            with open(file_path, "wb") as output_file:
                pkl.dump(metadata, output_file)
        else:
            self.users_dealitem_ids = metadata['gusers']
            self.dealitem_users_ids = metadata['gitems']
            self.top_n_items = metadata['top']
            self.coupons_in_time = metadata['coupons']

        prt("Fitting model finished successfully.")

    def predict(self, activities, y, top_N_items=top_N, distance_treshold=0.4, show_null_values=False, test_count=None):
        prt("Predicting...")

        user_i = 0
        for user in y:
            if test_count and user_i > test_count:
                break
            user_i = user_i + 1
            rec_top_n = False
            rec_top_n_w = False
            recommended_items = []

            # new user. recommend top items
            if str(user[0]) not in self.users_dealitem_ids:
                rec_top_n = True
                recommended_items = self.top_n_items
            else:
                user_items = self.users_dealitem_ids[str(user[0])]

                if len(user_items) < 1:
                    recommended_items = self.top_n_items
                    rec_top_n = True
                elif len(user_items) >= 1 and len(user_items) <= 2:
                    recommended_items = self.recommend_items(activities.loc[activities['create_time'] >= (self.actual_time - MONTH)], user_items, user[0],
                                                             distance_treshold=distance_treshold, top_from_user_city=True)
                    rec_top_n_w = True
                else:
                    recommended_items = self.recommend_items(activities, user_items, user[0],
                                                             distance_treshold=distance_treshold)

            if len(recommended_items) < 1:
                rec_top_n = True
                recommended_items = self.top_n_items

            u = recommended_items
            v = list(user[1].groupby('dealitem_id').groups.keys())

            y_pred, y_true = get_vectors(u, v)

            precision = precision_score(y_true, y_pred)

            ndcg_arr = np.zeros(len(recommended_items))
            for index, x in enumerate(recommended_items):
                if x in v:
                    ndcg_arr[index] = 1

            ndcg = ndcg_at_k(ndcg_arr, 5)
            mse = mean_squared_error(y_true, y_pred)

            hits = 0
            for rec_item in u:
                hits += v.count(rec_item)

            avg_hits = hits / len(v)

            if show_null_values or (hits > 0):
                if rec_top_n_w:
                    add_ndcg(self.ndcg_scores, ndcg, type='topw')
                    self.precision_scores_top_w.append(precision)
                    self.avg_hits_top_w.append(avg_hits)
                    print_report('TOP-BASED',precision, np.average(self.precision_scores_top_w), ndcg, mse, user[0], user_i, hits, np.average(self.avg_hits_top_w), v)
                elif rec_top_n:
                    add_ndcg(self.ndcg_scores, ndcg, type='topn')
                    self.precision_scores_top.append(precision)
                    self.avg_hits_top_n.append(avg_hits)
                    print_report('TOP-N\t',precision, np.average(self.precision_scores_top), ndcg, mse, user[0], user_i, hits, np.average(self.avg_hits_top_n), v)
                else:
                    add_ndcg(self.ndcg_scores, ndcg, type='def')
                    self.precision_scores_default.append(precision)
                    self.avg_hits_default.append(avg_hits)
                    print_report('DEFAULT\t',precision, np.average(self.precision_scores_default), ndcg, mse, user[0], user_i, hits, np.average(self.avg_hits_default), v)
            else:
                if rec_top_n:
                    self.precision_scores_top.append(precision)
                    self.avg_hits_top_n.append(avg_hits)
                    add_ndcg(self.ndcg_scores, ndcg, type='topn')

                elif rec_top_n_w:
                    self.precision_scores_top_w.append(precision)
                    self.avg_hits_top_w.append(avg_hits)
                    add_ndcg(self.ndcg_scores, ndcg, type='topw')

                else:
                    self.precision_scores_default.append(precision)
                    self.avg_hits_default.append(avg_hits)
                    add_ndcg(self.ndcg_scores, ndcg, type='def')


        prt("---- Statistics for (%d) users----" % user_i)
        # prt("users count | avg. default(%f)/%d | avg. top (%f)/%d" % (
        #     user_i, np.average(self.precision_scores_default), len(self.precision_scores_default),
        #     np.average(self.precision_scores_top), len(self.precision_scores_top)))

        prt("Avg hits - default: %f | top: %f | top based: %f" % (np.average(self.avg_hits_default), np.average(self.avg_hits_top_n),np.average(self.avg_hits_top_w)))
        prt("Avg ndcg-5 - default: %f | top: %f | top based: %f" % (np.average(self.ndcg_scores['def']), np.average(self.ndcg_scores['topn']),np.average(self.ndcg_scores['topw'])))

        prt("DEFAULT recommender\t at least one(%d / %d)" % ((len(self.avg_hits_default) - list(self.avg_hits_default).count(0)),len(self.avg_hits_default) ))
        prt("TOP-N recommender\t at least one(%d / %d)" % ((len(self.avg_hits_top_n) - list(self.avg_hits_top_n).count(0)),len(self.avg_hits_top_n)))
        prt("TOP-BASED recommender\t at least one(%d / %d)" % ((len(self.avg_hits_top_w) - list(self.avg_hits_top_w).count(0)), len(self.avg_hits_top_w) ))



    def recommend_items(self, activities, user_items, user, distance_treshold=0.5, top_from_user_items=False, top_from_user_city=False):
        N_items = top_N
        similar_user_ids = []
        user_city = ''

        cities_candidates = []
        for item in user_items:
            similar_user_ids.append(self.dealitem_users_ids[str(item)])
            if top_from_user_city:
                cities_users = self.grouped_by_items_ids.get_group(item)[['title_city', 'user_id']]
                cities_candidates.append(list(cities_users.loc[cities_users['user_id'] == float(user)]['title_city'])[0])

        distances = []
        similar_items = []
        for i in [val for sublist in similar_user_ids for val in sublist]:
            similar_items.append(self.users_dealitem_ids[str(i)])
            if i != user:
                u, v = get_vectors(self.users_dealitem_ids[str(i)], user_items)
                distances.append([i, sc.cosine(u, v)])

        recommended_items = []

        if top_from_user_city:
            ## top from city
            user_city = max(set(cities_candidates), key=cities_candidates.count)
            top_n_items, _ = get_top_n_and_coupons(self.full_data.loc[self.full_data['title_city'].str.lower() == str(user_city).lower()], self.deal_items, self.actual_time, top_N)
            return top_n_items

        if top_from_user_items:
            ## top from user's items
            items_count_map = {}
            for i in [val for sublist in similar_items for val in sublist]:
                count = 0
                if str(i) in items_count_map:
                    count = items_count_map[str(i)] + 1
                else:
                    count = 1
                items_count_map.update({str(i): count})

            for it in sorted(items_count_map.items(), key=operator.itemgetter(1), reverse=True)[:len(self.top_n_items) - 1]:
                recommended_items.append(it[0])

            return recommended_items

        # default recommender
        distances.sort(key=lambda t: t[1])
        self.distance_matrix.append({user: distances[:100]})

        for i in distances:
            top_items = self.users_dealitem_ids[str(i[0])]

            for item in top_items:
                # not in recommended items.
                # do not recommend already bought item
                if item not in recommended_items and i[1] <= distance_treshold:
                    if str(item) in self.coupons_in_time and self.coupons_in_time[str(item)]:
                        recommended_items.append(item)

                if (len(recommended_items) > N_items):
                    break

            if (len(recommended_items) > N_items):
                break

        return recommended_items
