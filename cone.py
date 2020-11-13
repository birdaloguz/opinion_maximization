from sgd import *
from greedy import *
import numpy as np
import networkx as nx

##get matrix
"""movie_ratings = pd.read_csv('datasets/CiaoDVD/movie-ratings.txt', names=["user_id", "item_id", "genre_id", "review_id", "rating", "timestamp"],
                header=None, sep=',', engine='python').drop_duplicates(subset=['user_id', 'item_id'])"""
"""movie_ratings = pd.read_csv('datasets/random_dataset/dataset.csv', names=["user_id", "item_id", "rating"],
                header=None, sep=',', engine='python')"""
"""movie_ratings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])"""
"""movie_ratings = pd.read_csv('datasets/epinions/ratings_data.txt', sep=' ', names=['user_id', 'item_id', 'rating'], nrows=173812)"""
"""movie_ratings = pd.read_csv('datasets/filmtrust/ratings.txt', sep=' ', names=['user_id', 'item_id', 'rating']).drop_duplicates(subset=['user_id', 'item_id'])"""
movie_ratings = pd.read_csv('datasets/rmat/rmatdatasetcopy_weight.csv', sep=',', names=['user_id', 'item_id', 'rating']).drop_duplicates(subset=['user_id', 'item_id'])
movie_ratings_truth = pd.read_csv('datasets/rmat/rmatdatasetcopy.csv', sep=',', names=['user_id', 'item_id', 'rating']).drop_duplicates(subset=['user_id', 'item_id'])

#average rating for opinion matrix
r_avg = movie_ratings['rating'].mean()

matrix_um = movie_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
matrix_um_truth = movie_ratings_truth.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

c = 0
hashmap = {}
reverse_hashmap = {}
for i in matrix_um.index.tolist():
    hashmap[c] = i
    reverse_hashmap[i] = c
    c += 1

#drop users with less than 5 ratings
droplist = [i for i in matrix_um.columns if np.count_nonzero(matrix_um[i])<5]
matrix_um.drop(droplist, axis=1, inplace=True)

droplist = [i for i in matrix_um_truth.columns if np.count_nonzero(matrix_um_truth[i])<5]
matrix_um_truth.drop(droplist, axis=1, inplace=True)

#2d array matrix
A = matrix_um.values
A_truth = matrix_um_truth.values


##load graph
G = nx.DiGraph()


#with open('datasets/CiaoDVD/trusts.txt') as fp:
#with open('datasets/random_dataset/dataset_trust.csv') as fp:
#with open('datasets/ml-100k/ml_trust.csv') as fp:
#with open('datasets/epinions/trust_data.txt') as fp:
#with open('datasets/filmtrust/trust.txt') as fp:
with open('datasets/rmat/rmat_trustcopy.txt') as fp:
    for idx, line in enumerate(fp.readlines()):
        a = line.split('\t')[0]
        b = line.split('\t')[1]
        G.add_edge(int(a), int(b))

from sklearn.metrics.pairwise import pairwise_distances
sim = (1-pairwise_distances(A.T, metric = "cosine"))*2

print(len(list(G.edges)))
for edge in list(G.edges):
    try:
        #jaccard = sim[reverse_hashmap[edge[0]]][reverse_hashmap[edge[1]]]
        jaccard = len(list(set(G.predecessors(edge[0])) & set(G.successors(edge[1])))) / len(
                    list(set(G.predecessors(edge[0])) | set(G.successors(edge[1]))))   
    except:
        jaccard = 0
    if jaccard <= 0:
        G.remove_edge(edge[0], edge[1])
    else:
        G[edge[0]][edge[1]]['weight'] = jaccard

print(len(list(G.edges)))

for node in list(G.nodes):
    G.nodes[node]['threshold'] = round(random.uniform(0, 1), 3)



print(-1)
#initialization -- fill matrix
U, V, val, val_ratings, val_ratings_truth = get_rating_estimations(A, A_truth, validation=True)


r_avg = (np.dot(U, V.T)-np.dot(U, V.T).mean())*2+np.dot(U, V.T).mean()
r_avg[r_avg<1]=1
r_avg[r_avg>5]=5
r_avg = r_avg.mean()
print(val_ratings.mean())
print(r_avg)

r_avg_truth = A_truth.mean() 
r_avg_truth_pt = val_ratings_truth.mean()
print(r_avg_truth)
print(r_avg_truth_pt)
test=0
avg = 0
for i in list(G.nodes):
    if val_ratings[reverse_hashmap[i]] > 5:
        avg+=5
        test+=5-r_avg
    elif val_ratings[reverse_hashmap[i]] < 1:
        avg+=1
        test+=1-r_avg
    else:
        avg+=val_ratings[reverse_hashmap[i]]
        test+=val_ratings[reverse_hashmap[i]]-r_avg
print(avg/len(list(G.nodes)))
print(test)
print(len(list(G.nodes)))

#delete validation item
A = np.delete(A, np.s_[val], axis=1)

##init random product vector v_t
v_t = np.random.uniform(1, 1.5, size=(1, 3))
V = np.append(V, v_t, axis=0)

C_p = [] #activated nodes
T = 50 #rounds
seed_set = []
opinion_set = []

import time

res = {"org": {"positive": {},
                "negative": {}},
        "truth": {"positive": {},
                "negative": {}},
        "truth_pt": {
                "positive": {},
                "negative": {}}}
spread =[]

candidate_neighbors = list(G.nodes)

from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=8, metric='cosine').fit(U)

u_clusters=[[] for i in range(8)]
for idx,r in enumerate(kmedoids.labels_):
    u_clusters[r].append(hashmap[idx])

init_seeds = [random.sample(i, 3) for i in u_clusters]
time_set=[]
for t in range(10):
    print("Round: "+str(t))
    
    start = time.time()
    v_t = V[-1] #new product profile
    
    r = (np.dot(U, v_t.T)-np.dot(U, v_t.T).mean())*2+np.dot(U, v_t.T).mean() #new product ratings
    
    r_q = r*1 #new product opinions
    r_q[r_q<1]=1
    r_q[r_q>5]=5
    R_new_p = (np.dot(U, V.T)-np.dot(U, V.T).mean())*2+np.dot(U, V.T).mean() #approximate ratings matrix

    r_q = r_q - R_new_p.mean()
    

    if t!=0:
        S_k, C_p, O_max = greedy_om(G, r_q, t, reverse_hashmap, candidate_neighbors, C_p=C_p)
    else:
        O_max = 0
        S_k=[]
        for cluster in init_seeds:
            for clus in cluster:
                S_k.append(clus)
        C_p=S_k
        O_clus_max = float('-inf')
        max_clus_id = -1
        for clus_id, cluster in enumerate(init_seeds):
            O_clus=0
            for clus in cluster:
                O_clus+= r_q[reverse_hashmap[clus]]
            if O_clus>=O_clus_max:
                O_clus_max=O_clus
                max_clus_id=clus_id
        candidate_neighbors=u_clusters[max_clus_id]

    seed_set.append(S_k)
    print(S_k)
    opinion_set.append(O_max)
    C_p.sort()
    print("Activated nodes")
    print(C_p)
    print(len(C_p))
    spread.append(len(C_p))
    
    ##profile update
    r_v = 1*val_ratings_truth#val_ratings


    updated_A = np.hstack((A, np.reshape(r_v, (len(r_v), 1))))
    
    U, V = get_rating_estimations(updated_A, A_truth)

    elapsed = time.time() - start
    print("Time elapsed for the round: " + str(elapsed))
    time_set.append(elapsed)

    positive = 0
    negative= 0
    
    for c in C_p:
        if val_ratings[reverse_hashmap[c]]>=5:
            positive+=5-r_avg
        elif val_ratings[reverse_hashmap[c]]<=1:
            negative+=1-r_avg
        else:
            if (val_ratings[reverse_hashmap[c]]-r_avg)<0:
                negative+=val_ratings[reverse_hashmap[c]]-r_avg
            else:
                positive+=val_ratings[reverse_hashmap[c]]-r_avg
    print("Results: ")
    print("Positive: "+str(positive))
    print("Negative: "+str(negative))
    print("Total Opinion: "+str(positive+negative))
    res["org"]["positive"][t]=positive
    res["org"]["negative"][t]=negative
    #print(res)
    positive = 0
    negative= 0
    
    for c in C_p:
        if (val_ratings_truth[reverse_hashmap[c]]-r_avg_truth)<0:
            negative+=val_ratings_truth[reverse_hashmap[c]]-r_avg_truth
        else:
            positive+=val_ratings_truth[reverse_hashmap[c]]-r_avg_truth
    print("Results: ")
    print("Positive: "+str(positive))
    print("Negative: "+str(negative))
    print("Total Opinion: "+str(positive+negative))
    res["truth"]["positive"][t]=positive
    res["truth"]["negative"][t]=negative
    #print(res)
    positive = 0
    negative= 0
    
    for c in C_p:
        if (val_ratings_truth[reverse_hashmap[c]]-r_avg_truth_pt)<0:
            negative+=val_ratings_truth[reverse_hashmap[c]]-r_avg_truth_pt
        else:
            positive+=val_ratings_truth[reverse_hashmap[c]]-r_avg_truth_pt
    print("Results: ")
    print("Positive: "+str(positive))
    print("Negative: "+str(negative))
    print("Total Opinion: "+str(positive+negative))
    res["truth_pt"]["positive"][t]=positive
    res["truth_pt"]["negative"][t]=negative
    print(spread)

print(res)
print(time_set)