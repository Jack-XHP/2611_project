# 2611_project


## DWUG BERT embedding

```
python embed.py
```
## DWUG balancing and plot balance factor histogram
```
python balance_cluster.py
```


## unsupervised training for MinCut or DMoN

```
python train.py --model [MinCut|DMoN] --layer [GCN layers]  --balance [rebalance DWUG]
```

## Kmeans clusters

```
python kmean.py --balance [rebalance DWUG]
```

## plot figures
after training kmean, mincut, dmon on both balanced and unbalanced DWUG

```
python plot.py
```
