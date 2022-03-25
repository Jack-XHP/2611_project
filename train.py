import pickle

import torch
from sklearn.metrics import normalized_mutual_info_score

from DWUG import DWUG
from graph_cluster import MinCut, pairwise_recall, pairwise_accuracy, DMON

dataset = DWUG("./dwug_en/graphs/bert_balance")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DMON(dataset.num_features).to(device)
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
    NMI = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
    acc = pairwise_accuracy(y, y_pred)
    recall = pairwise_recall(y, y_pred)
    F1 = 2 * acc * recall / (acc + recall)
    return tot_loss.cpu().item(), NMI, F1


print("Start training...")
results = []
for j in range(len(dataset)):
    word_result = []
    for i in range(1000):
        train_loss = train(dataset[j])
        test_loss, NMI, F1 = test(dataset[j])
        word_result.append([i, train_loss, test_loss, NMI, F1])
        # print(f"epoch {i}  train_loss {train_loss}  test_loss {test_loss}  test NMI {NMI} test F1 {F1}")
    results.append(word_result)
with open(r"dmonv2.pickle", "wb") as output_file:
    pickle.dump(results, output_file)
