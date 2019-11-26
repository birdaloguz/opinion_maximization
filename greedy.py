import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import random


def greedy_om(G, r_q, C_p = [], k=50):
    S=[] #seeds for current round
    while list(set(list(G.nodes))-set(list(set().union(C_p ,S))))!=[] and len(S)!=k:
        O_max = float("-inf")
        for user in list(set(list(G.nodes))-set(list(set().union(C_p ,S)))):
            ##find total opinion for each user O and find activated nodes C
            S_t = list(set().union(C_p ,S, [user]))
            O=0
            C=[]
            infectee_dict = {}
            for s in S_t:
                C.append(s)
                infectees = list(G.successors(s))
                for i in infectees:
                    try:
                        infectee_dict[i] += G[s][i]['weight']
                    except:
                        infectee_dict[i] = G[s][i]['weight']

                    if infectee_dict[i] > G.nodes[i]['threshold']:
                        O+=r_q[i-1][0]
                        C.append(i)
            #Calculate O_max
            if O>O_max:
                O_max = O
                user_best = user
                C_best = C
        S = list(set().union(S, [user_best]))
        C_out = C_best
    return S, C_out

