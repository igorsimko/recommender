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


def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        if float(best) == 0:
            best = 1
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

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
