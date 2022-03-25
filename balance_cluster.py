import os
import numpy as np
import networkx as nx


directory = "./dwug_en/graphs/bert/raw"
for file in os.listdir(directory):
    graph = nx.read_gpickle(f"{directory}/{file}")
    y = np.array([graph.nodes[node]['cluster'] for node in list(graph.nodes)])
    if y.max() == 0:
        continue   
    count, bin = np.histogram(y, bins=range(y.max()+2))
    max_count = count[0]
    for val in bin[1:-1]:
        cluster_name = [list(graph.nodes)[i] for i in np.argwhere(y==val).flatten()]
        cluster_node = [graph.nodes[node] for node in cluster_name]
        for i in range(max_count//count[val]):
            cluster_new_name = [str(i)+name for name in cluster_name]
            graph.add_nodes_from(cluster_new_name)
            for node in cluster_name:
                for key in graph.nodes[cluster_name[0]].keys():
                    graph.nodes[str(i)+node][key] = graph.nodes[node][key]
            cluster_edges = [graph.edges(node) for node in cluster_name]
            cluster = set(cluster_name)
            cluster_edges_mod = [[(str(i)+n1, str(i)+n2 if n2 in cluster else n2, graph[n1][n2]['weight']) for (n1, n2) in edges] for edges in cluster_edges]
            new_edges = [(cluster_name[j], cluster_new_name[j], 4.0) for j in range(len(cluster_name))]
            cluster_edges_mod.append(new_edges)
            for edges in cluster_edges_mod:
                for n1, n2, weight in edges:
                    graph.add_edge(n1, n2, weight=weight)   
    nx.write_gpickle(graph, f"./dwug_en/graphs/bert_balance/raw/{file}")
    y = np.array([graph.nodes[node]['cluster'] for node in list(graph.nodes)]) 
    count, bin = np.histogram(y, bins=range(y.max()+2))
    print(count)
    print(bin)