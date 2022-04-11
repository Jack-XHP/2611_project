import pickle

import torch
from sklearn.metrics import normalized_mutual_info_score, rand_score

from DWUG import DWUG
from graph_cluster import MinCut, pairwise_recall, pairwise_accuracy, DMON
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, required=True, default="DMoN")
parser.add_argument('--layer', type=int, required=True, default=1)
parser.add_argument("--balance", default=False, action="store_true")
parser.add_argument('--device', type=int, required=True, default=0)
args = parser.parse_args()
if args.balance:
    dataset = DWUG("./dwug_en/graphs/bert_balance")
else:
    dataset = DWUG("./dwug_en/graphs/bert")
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
if args.model == "MinCut":
    model = MinCut(dataset.num_features, layer=args.layer).to(device)
else:
    model = DMON(dataset.num_features, layer=args.layer).to(device)
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
    rand = rand_score(y, y_pred)
    return tot_loss.cpu().item(), NMI, F1, rand


print("Start training...")
results = []
for j in range(len(dataset)):
    word_result = []
    for i in range(1000):
        train_loss = train(dataset[j])
        test_loss, NMI, F1, rand = test(dataset[j])
        word_result.append([i, train_loss, test_loss, NMI, F1, rand])
        # print(f"epoch {i}  train_loss {train_loss}  test_loss {test_loss}  test NMI {NMI} test F1 {F1}")
    results.append(word_result)
with open(rf"{args.model}{args.layer}{int(args.balance)}.pickle", "wb") as output_file:
    pickle.dump(results, output_file)
