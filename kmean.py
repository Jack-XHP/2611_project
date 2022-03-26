
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, rand_score
import os
import numpy as np
import networkx as nx
from graph_cluster import MinCut, pairwise_recall, pairwise_accuracy, DMON
import pickle
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--balance", default=False, action="store_true")
args = parser.parse_args()
if args.balance:
    directory = "./dwug_en/graphs/bert_balance/raw"
else:
    directory = "./dwug_en/graphs/bert/raw"
results = []
for file in os.listdir(directory):
    graph = nx.read_gpickle(f"{directory}/{file}")
    for n1, n2 in graph.edges:
        graph[n1][n2]['weight'] = np.nan_to_num(graph[n1][n2]['weight'])
    y = np.array([graph.nodes[node]['cluster'] for node in list(graph.nodes)])
    ev = nx.linalg.algebraicconnectivity.fiedler_vector(graph, weight='weight')
    kmeans = KMeans(n_clusters=y.max()+1, random_state=0).fit(ev.reshape(-1,1))
    NMI = normalized_mutual_info_score(y, kmeans.labels_, average_method='arithmetic')
    rand = rand_score(y, kmeans.labels_)
    acc = pairwise_accuracy(y, y)
    recall = pairwise_recall(y, y)
    F1 = 2 * acc * recall / (acc + recall)
    results.append((file, NMI, F1, rand))
with open(rf"kmean{int(args.balance)}.pickle", "wb") as output_file:
    pickle.dump(results, output_file)