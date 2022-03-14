import os.path as osp
import os
import networkx as nx
from torch_geometric.data import Dataset, Data, download_url
import torch
import numpy as np


class DWUG(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(44)]

    def process(self):
        idx = 0
        for file in os.listdir(self.raw_dir):
            # Read data from `raw_path`.
            graph = nx.read_gpickle(f"{self.raw_dir}/{file}")
            x = torch.tensor([graph.nodes[node]['bert_embedding'][graph.nodes[node]['bert_index']]for node in  list(graph.nodes)], dtype=torch.float)
            nodeslist = [node for node in  list(graph.nodes)]
            edge_index1 = [nodeslist.index(edge[0]) for edge in graph.edges]
            edge_index2 = [nodeslist.index(edge[1]) for edge in graph.edges]
            edge_index = torch.tensor([edge_index1 + edge_index2, edge_index2+edge_index1], dtype=torch.long)
            edge_weight = np.nan_to_num(np.array([[edge[2]['weight']] for edge in graph.edges.data()]))
            edge_weight = torch.tensor(edge_weight.tolist() + edge_weight.tolist(), dtype=torch.float)
            y = torch.tensor([graph.nodes[node]['cluster']for node in  list(graph.nodes)], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

