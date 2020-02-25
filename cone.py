from sgd import *
from greedy import *
import numpy as np
import networkx as nx

##get matrix
"""movie_ratings = pd.read_csv('CiaoDVD/movie-ratings.txt', names=["user_id", "movie_id", "genre_id", "review_id", "rating", "timestamp"],
                header=None, sep=',', engine='python').drop_duplicates(subset=['user_id', 'movie_id'])"""
movie_ratings = pd.read_csv('random_dataset/dataset.csv', names=["user_id", "item_id", "rating"],
                header=None, sep=',', engine='python')

#average rating for opinion matrix
r_avg = movie_ratings['rating'].mean()
matrix_um = movie_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

#drop users with less than 5 ratings
droplist = [i for i in matrix_um.columns if np.count_nonzero(matrix_um[i])<5]
matrix_um.drop(droplist, axis=1, inplace=True)

#2d array matrix
A = matrix_um.values

##load graph

G = nx.DiGraph()
with open('random_dataset/dataset_trust.csv') as fp:
    for idx, line in enumerate(fp.readlines()):
        a = line.split(',')[0]
        b = line.split(',')[1]
        G.add_edge(int(a), int(b))

for edge in list(G.edges):
    try:
        jaccard = len(list(set(G.predecessors(edge[0])) & set(G.successors(edge[1])))) / len(
                    list(set(G.predecessors(edge[0])) | set(G.successors(edge[1]))))
    except:
        jaccard = 0
    if jaccard == 0:
        G.remove_edge(edge[0], edge[1])
    else:
        G[edge[0]][edge[1]]['weight'] = jaccard

for node in list(G.nodes):
    G.nodes[node]['threshold'] = round(random.uniform(0, 1), 3)
print(-1)
#initialization -- fill matrix
R_tilda, opinion_matrix, U, V, val, val_ratings= get_rating_estimations(A, validation=True)
#delete validation item
A = np.delete(A, np.s_[val], axis=1)

##init random product vector v_t
v_t = np.random.uniform(0, 5, size=(1, 3))
V = np.append(V, v_t, axis=0)


C_p = [] #activated nodes
T = 50 #rounds
seed_set = []
opinion_set = []

import time
for t in range(T):
    print(t)
    start = time.time()
    v_t = V[-1] #new product profile
    r = np.dot(U, v_t.T) #new product ratings

    r_q = r - r_avg #new product opinions
    R_new_p = np.dot(U, V.T) #ratings matrix

    S_k, C_p, O_max = greedy_om(G, r_q, C_p=C_p)

    seed_set.append(S_k)
    opinion_set.append(O_max)
    C_p.sort()

    ##profile update
    r_v = R_new_p[:, -1]
    for idx, i in enumerate(r_q):
        if idx+1 not in C_p:
            r_v[idx]=0

    updated_A = np.hstack((A, np.reshape(r_v, (len(r_v), 1))))
    R_tilda, opinion_matrix, U, V = get_rating_estimations(updated_A)
    elapsed = start - time.time()
    print(elapsed)

print(seed_set)
print(val)
print(list(val_ratings))
print(r_avg)
print(C_p)
np.savetxt("val_ratings.txt", val_ratings, newline=" ")


