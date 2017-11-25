import numpy as np
import datetime


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


def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))
