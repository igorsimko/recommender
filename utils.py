import numpy as np
import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

MONTH = 2678400

def get_vectors(A, B):
    AB = list(set(A + B))
    u = np.zeros(len(AB), dtype=int)
    v = np.zeros(len(AB), dtype=int)

    for i in range(len(AB)):
        if AB[i] in B and AB[i] in A:
            u[i] = 1
            v[i] = 1
        if AB[i] in B and AB[i] not in A:
            u[i] = 0
            v[i] = 1
        if AB[i] not in B and AB[i] in A:
            u[i] = 1
            v[i] = 0

    return u, v


def valid_time_for_item(actual_time, min_time, max_time):
    return actual_time <= max_time and actual_time >= min_time


def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))


def print_report(recommender, precision, global_precision, ndcg, mse, user, i, hits, avg_hits, y_true):
    prt(
        "%s\tUser [%d] with precision(%f), global precision(%f), ndcg(%f), mse(%f), hits(%d/%d), global hits(%f) of %d users " % (
            recommender, user, precision, global_precision, ndcg, mse, hits, len(y_true), avg_hits, i))

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def get_top_n_and_coupons(full_data, deal_items, actual_time, top_n):
    top_n_items = []
    coupons_in_time = {}

    sorted_list = full_data.loc[full_data['create_time'] >= (actual_time - MONTH)].groupby('dealitem_id')
    # sort deals
    for top in sorted(sorted_list, key=lambda x: len(x[1]), reverse=True):
        deal = deal_items.loc[deal_items['id'] == float(top[0])]
        if deal.empty == False:
            max_time = deal['coupon_end_time'].values[0]
            min_time = deal['coupon_begin_time'].values[0]

            valid_time = valid_time_for_item(actual_time, min_time, max_time)
            if valid_time and len(top_n_items) < top_n:
                top_n_items.append(top[0])

            coupons_in_time.update({str(top[0]): valid_time})

    return top_n_items, coupons_in_time


def add_ndcg(ndcg_arr, value, type='def'):
    ndcg_list = []
    if type in ndcg_arr:
        ndcg_list = list(ndcg_arr[type])
        ndcg_list.append(value)

    else:
        ndcg_list.append(value)

    ndcg_arr.update({type: ndcg_list})