
import torch
from torch.nn import Linear, Module
from torch.autograd import Variable
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DMoNPooling, GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import cluster, normalized_mutual_info_score
import numpy as np
from DWUG import DWUG
import pickle


dataset = DWUG("./dwug_en/graphs/bert")


class DMON(Module):
    def __init__(self, in_channels, hidden_channels=64, num_nodes=16, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes, dropout)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        adj = to_dense_adj(edge_index)
        cluster, x, adj, spectral_loss, ortho_loss, cluster_loss = self.pool1(x, adj)
        return cluster, spectral_loss + ortho_loss + cluster_loss


class MinCut(Module):
    def __init__(self, in_channels, hidden_channels=64, num_nodes=16, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = Linear(hidden_channels, num_nodes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        adj = to_dense_adj(edge_index)
        cluster = self.pool1(x)
        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, cluster)
        return [torch.softmax(cluster, dim=-1)],  ortho_loss + mincut_loss


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MinCut(dataset.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(data):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out, tot_loss = model(data.x, data.edge_index, data.edge_attr)
    tot_loss.backward()
    optimizer.step()
    return tot_loss.cpu().item()


@torch.no_grad()
def test(data):
    model.eval()
    data = data.to(device)
    pred, tot_loss = model(data.x, data.edge_index, data.edge_attr)
    y_pred = pred[0].argmax(dim=1).cpu().numpy()
    y = data.y.cpu().numpy()
    NMI =  normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
    acc = pairwise_accuracy(y, y_pred)
    recall = pairwise_recall(y, y_pred)
    F1 = 2 *  acc * recall / (acc+recall)
    return tot_loss.cpu().item(), NMI, F1

results = []
for j in range(44):
    word_result = []
    for i in range(1000):
        train_loss = train(dataset[j])
        test_loss, NMI, F1 = test(dataset[j])
        word_result.append([i, train_loss, test_loss, NMI, F1])
        # print(f"epoch {i}  train_loss {train_loss}  test_loss {test_loss}  test NMI {NMI} test F1 {F1}")
    results.append(word_result)
print(results)
with open(r"mincut.pickle", "wb") as output_file:
    pickle.dump(results, output_file)