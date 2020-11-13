def greedy_om(G, r_q, t, reverse_hashmap, candidate_neighbors, C_p = [], k=1):
    S=[] #seeds for current round
    C_out=[]
    C_best=[]
    O_max = float("-inf")
    while list(set(list(G.nodes))-set(list(set().union(C_p ,S))))!=[] and len(S)!=k:
        O_max = float("-inf")       
        for user in list(set(candidate_neighbors)-set(list(set().union(C_p ,S)))):
            ##find total opinion for each user O and find activated nodes C
            S_t = list(set().union(C_p ,S, [user]))
            O=0
            C=[]
            infectee_dict = {}
            for s in S_t:
                O+=r_q[reverse_hashmap[s]]
                C.append(s)
                infectees = list(set(list(G.successors(s)))-set(C_p)-set(C))
                for i in infectees:
                    if i not in C_p and i not in C:
                        try:
                            infectee_dict[i] += G[s][i]['weight']
                        except:
                            infectee_dict[i] = G[s][i]['weight']

                        if infectee_dict[i] > G.nodes[i]['threshold']:
                            O+=r_q[reverse_hashmap[i]]
                            C.append(i)
            #Calculate O_max
            if O>O_max:
                O_max = O
                user_best = user
                C_best = C
                
        S = list(set().union(S, [user_best]))
        C_out = C_best
        

    return S, C_out, O_max


