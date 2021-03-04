import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def subset(iterable):
    
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def skeleton(suffStat, indepTest, alpha, labels, method,
             fixedGaps, fixedEdges,
             NAdelete, m_max, numCores, verbose):

    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        G[i][i] = False

    done = False

    ord = 0
    n_edgetests = {0: 0}
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0

        done = True

        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y] == True:
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = False

                neighbors = [i for i in range(len(neighborsBool)) if neighborsBool[i] == True]

                if len(neighbors) >= ord:

                    if len(neighbors) > ord:
                        done = False

                    for neighbors_S in set(itertools.combinations(neighbors, ord)):

                        n_edgetests[ord1] = n_edgetests[ord1] + 1

                        pval = indepTest(suffStat, x, y, list(neighbors_S))

                        if pval >= alpha:
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = list(neighbors_S)
                            break

        ord += 1

    return {'sk': np.array(G), 'sepset': sepset}

def extend_cpdag(graph):
    def rule1(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)
            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2
        return pdag

    def rule2(pdag, solve_conf=False):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2
        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[1],x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        # unfaithful
                        if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break
        return pdag

    pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))


    for x, y in sorted(ind, key=lambda x:(x[1],x[0])):
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and z != x:
                allZ.append(z)

        for z in allZ:
            if graph['sk'][x][z] == False and graph['sepset'][x][z] != None and graph['sepset'][z][x] != None and not (
                    y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest, p='Use labels',
       fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float("inf"),
       u2pd=("relaxed", "rand", "retry"),
       skel_method=("stable", "original", "stable.fast"),
       conservative=False, maj_rule=False, solve_confl=False,
       numCores=1, verbose=False):

    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)

    return extend_cpdag(graphDict)

def gaussCItest(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]

    cut_at = 0.9999999

    if len(S) == 0:
        r = C[x, y]

    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))

    else:
        m = C[np.ix_([x]+[y]+S, [x]+[y]+S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1*cut_at, r))

    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))

    return 2 * (1 - norm.cdf(abs(res)))

def Dfs(graph, k, path, vis):
    flag = False
    for i in range(len(graph)):
        if (graph[i][k] == 1) and (vis[i] != True) :
            
            flag =True
            vis[i] = True
            path.append(i)
            Dfs(graph, i, path, vis)
            path.pop()
            vis[i] = False

    if flag == False:
        print(path)

def Draw(graph, labels):

    G = nx.DiGraph()

    for i in range(len(graph)):
        G.add_node(labels[i])
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edges_from([(labels[i], labels[j])])

    nx.draw(G, with_labels=True)
    plt.savefig("result.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./input.txt')
    args = parser.parse_args()
    train_filename = args.train_dir
    data = []
    with open(train_filename, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            line = list(line.strip().split())
            for item in line:
                item = item.split('_')
                s = []
                for i, val in enumerate(item):
                    if i==1:
                        s.append(val)
                    else:
                        s.append(eval(val))
                data.append(s)
    train_data = pd.DataFrame(data, columns=['厚度','材质','质量'])
    train_data = pd.get_dummies(train_data)
    x = train_data.columns.values.tolist()
    x = np.array(x)
    x[[1,-1]] = x[[-1, 1]]

    feature_dim = train_data.shape[1]
    
    train_data = train_data.values
    train_data[:,[1,-1]] = train_data[:,[-1,1]]
    
    
    dict_data = {}
    for idx in range(feature_dim):
        key = 'x' + str(idx)
        dict_data[key] = train_data[:, idx]
    data = pd.DataFrame(dict_data)
    row_count = sum(1 for row in data)
    labels = [i for i in range(row_count)]
    p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05, labels=[str(i) for i in range(row_count)], indepTest=gaussCItest)
    cause = []
    for idx in range(row_count):
    	row_ = p[idx]
    	if row_[-1] == 1:
    		cause.append(idx)
    result_list = open('./output.txt', 'w')
    for j in cause:
        result_list.write(x[j]+' ')
    result_list.close()
    Draw(p, labels)