import numpy as np
import torch
from sklearn.metrics import cluster
from torch.nn import Linear, Module
from torch_geometric.nn import DMoNPooling, GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj


class DMON(Module):
    def __init__(self, in_channels, hidden_channels=64, num_nodes=16, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes, dropout)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        #x = self.conv2(x, edge_index, edge_weight).relu()
        adj = to_dense_adj(edge_index)
        cluster, x, adj, spectral_loss, ortho_loss, cluster_loss = self.pool1(x, adj)
        return cluster, spectral_loss + ortho_loss + cluster_loss


class MinCut(Module):
    def __init__(self, in_channels, hidden_channels=64, num_nodes=16):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool1 = Linear(hidden_channels, num_nodes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        #x = self.conv2(x, edge_index, edge_weight).relu()
        adj = to_dense_adj(edge_index)
        cluster = self.pool1(x)
        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, cluster)
        return [torch.softmax(cluster, dim=-1)], ortho_loss + mincut_loss


def pairwise_recall(y_true, y_pred):
    true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def pairwise_accuracy(y_true, y_pred):
    true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
    return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)


def _pairwise_confusion(y_true, y_pred):
    contingency = cluster.contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (total - 1) - true_positives - false_positives - false_negatives
    return true_positives, false_positives, false_negatives, true_negatives
