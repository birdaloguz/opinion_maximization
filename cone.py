from sgd import *
from greedy import *
import numpy as np
import networkx as nx

##get matrix
movie_ratings = pd.read_csv('datasets/CiaoDVD/movie-ratings.txt', names=["user_id", "item_id", "genre_id", "review_id", "rating", "timestamp"],
                header=None, sep=',', engine='python').drop_duplicates(subset=['user_id', 'item_id'])
"""movie_ratings = pd.read_csv('datasets/random_dataset/dataset.csv', names=["user_id", "item_id", "rating"],
                header=None, sep=',', engine='python')"""
"""movie_ratings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])"""
"""movie_ratings = pd.read_csv('datasets/epinions/ratings_data.txt', sep=' ', names=['user_id', 'item_id', 'rating'], nrows=173812)"""
"""movie_ratings = pd.read_csv('datasets/filmtrust/ratings.txt', sep=' ', names=['user_id', 'item_id', 'rating']).drop_duplicates(subset=['user_id', 'item_id'])"""
"""movie_ratings = pd.read_csv('datasets/rmat/rmatdataset_2.csv', sep=',', names=['user_id', 'item_id', 'rating']).drop_duplicates(subset=['user_id', 'item_id'])
"""
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

with open('datasets/CiaoDVD/trusts.txt') as fp:
#with open('datasets/random_dataset/dataset_trust.csv') as fp:
#with open('datasets/ml-100k/ml_trust.csv') as fp:
#with open('datasets/epinions/trust_data.txt') as fp:
#with open('datasets/filmtrust/trust.txt') as fp:
#with open('datasets/rmat/rmat_trust.txt') as fp:
    for idx, line in enumerate(fp.readlines()):
        a = line.split(',')[0]
        b = line.split(',')[1]
        if int(a)<A.shape[0] and int(b)<A.shape[0]:
            G.add_edge(int(a), int(b))
print(len(list(G.edges)))
for edge in list(G.edges):
    try:
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
U, V, val, val_ratings = get_rating_estimations(A, validation=True)


r_avg = np.dot(U, V.T)
r_avg[r_avg<1]=1
r_avg[r_avg>5]=5
r_avg = r_avg.mean()
print(val_ratings.mean())
print(r_avg)

test=0
avg = 0
for i in list(G.nodes):
    if val_ratings[i-1] > 5:
        avg+=5
        test+=5-r_avg
    elif val_ratings[i-1] < 1:
        avg+=1
        test+=1-r_avg
    else:
        avg+=val_ratings[i-1]
        test+=val_ratings[i-1]-r_avg
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

for t in range(50):
    print("Round: "+str(t))
    f = open("results.txt", "a+")
    
    start = time.time()
    v_t = V[-1] #new product profile
    
    r = np.dot(U, v_t.T) #new product ratings
    
    r_q = r*1 #new product opinions
    r_q[r_q<1]=1
    r_q[r_q>5]=5
    R_new_p = np.dot(U, V.T) #approximate ratings matrix

    r_q = r_q - R_new_p.mean()
    
    

    S_k, C_p, O_max = greedy_om(G, r_q, t, C_p=C_p)
    seed_set.append(S_k)
    opinion_set.append(O_max)
    C_p.sort()
    print("Activated nodes")
    print(C_p)
    print(len(C_p))
    
    
    ##profile update
    r_v = 1*val_ratings

    for idx, i in enumerate(r_q):
        if idx+1 not in C_p:
            r_v[idx]=0

    updated_A = np.hstack((A, np.reshape(r_v, (len(r_v), 1))))
    
    U, V = get_rating_estimations(updated_A)
    
    elapsed = time.time() - start
    print("Time elapsed for the round: " + str(elapsed))

    positive = 0
    negative= 0
    
    for c in C_p:
        if val_ratings[c-1]>=5:
            positive+=5-r_avg
        elif val_ratings[c-1]<=1:
            negative+=1-r_avg
        else:
            if (val_ratings[c-1]-r_avg)<0:
                negative+=val_ratings[c-1]-r_avg
            else:
                positive+=val_ratings[c-1]-r_avg
    print("Results: ")
    print("Positive: "+str(positive))
    print("Negative: "+str(negative))
    print("Total Opinion: "+str(positive+negative))
    f.write(str(positive+negative)+"\n")
    f.close()


"""print(seed_set)
print(val)
print(list(val_ratings))
print(r_avg)
print(C_p)"""



