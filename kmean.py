
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import os
import numpy as np
import networkx as nx
from graph_cluster import MinCut, pairwise_recall, pairwise_accuracy, DMON
import pickle
directory = "./dwug_en/graphs/bert_balance/raw"
results = []
for file in os.listdir(directory):
    graph = nx.read_gpickle(f"{directory}/{file}")
    for n1, n2 in graph.edges:
        graph[n1][n2]['weight'] = np.nan_to_num(graph[n1][n2]['weight'])
    y = np.array([graph.nodes[node]['cluster'] for node in list(graph.nodes)])
    ev = nx.linalg.algebraicconnectivity.fiedler_vector(graph, weight='weight')
    kmeans = KMeans(n_clusters=y.max()+1, random_state=0).fit(ev.reshape(-1,1))
    NMI = normalized_mutual_info_score(y, kmeans.labels_, average_method='arithmetic')
    acc = pairwise_accuracy(y, kmeans.labels_)
    recall = pairwise_recall(y, kmeans.labels_)
    F1 = 2 * acc * recall / (acc + recall)
    print((file, NMI, F1))
    results.append((file, NMI, F1))
with open(r"kmean", "w") as output_file:
    pickle.dump(results, output_file)