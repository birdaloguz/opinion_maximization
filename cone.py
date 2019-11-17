from sgd import *
from greedy import *
import numpy as np


##get matrix
movie_ratings = pd.read_csv('CiaoDVD/movie-ratings.txt', names=["user_id", "movie_id", "genre_id", "review_id", "rating", "timestamp"],
                header=None, sep=',', engine='python').drop_duplicates(subset=['user_id', 'movie_id'])

matrix_um = movie_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
droplist = [i for i in matrix_um.columns if np.count_nonzero(matrix_um[i])<20]
matrix_um.drop(droplist, axis=1, inplace=True)
A = matrix_um.values

##load graph
G = nx.DiGraph()
with open('CiaoDVD/trusts.txt') as fp:
    for idx, line in enumerate(fp.readlines()):
        a = line.split(',')[1]
        b = line.split(',')[0]
        if int(a) <= 17615 and int(b) <= 17615:
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


R_tilda, opinion_matrix, U, V, r_avg = get_rating_estimations(A)

##init random product vector v_t
v_t = np.random.uniform(0, 2, size=(3, 1))
V = np.append(V, v_t, axis=1)


##greedy algorithm
v_new = np.random.rand(2,1)
C_p = []
T = 2 #rounds
seed_set = []

for t in range(T):
    r_q = np.dot(U.T, v_t) - r_avg
    R_new_p = np.dot(U.T, V)

    S_k, C_p = greedy_om(G, r_q, C_p=C_p)
    seed_set.append(S_k)
    C_p.sort()

    ##profile update
    r_v = R_new_p[:, -1]
    for idx, i in enumerate(r_q):
        if idx+1 not in C_p:
            r_v[idx]=0
    updated_A = np.hstack((A, np.reshape(r_v, (len(r_v), 1))))

    R_tilda, opinion_matrix, U, V, r_avg = get_rating_estimations(updated_A)


print(seed_set)


