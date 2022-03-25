import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
with open(r"mincutv2.pickle", "rb") as output_file:
    mincut = np.array(pickle.load(output_file))
with open(r"dmonv2.pickle", "rb") as output_file:
    dmon = np.array(pickle.load(output_file))



directory = "./dwug_en/graphs/bert_balance/raw"
words = [file.split('_')[0] for file in os.listdir(directory)]
mincut1 = [[words[i]] + list(mincut[i, -1, 3:]) for i in range(len(words))]
dmon1 = [[words[i]] + list(dmon[i, -1, 3:]) for i in range(len(words))]
print(mincut1)
print(dmon1)
[plt.plot(mincut[i, :, 1], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()
[plt.plot(mincut[i, :, 2], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(mincut[i, :, 3], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(mincut[i, :, 4], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(dmon[i, :, 1], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()
[plt.plot(dmon[i, :, 2], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(dmon[i, :, 3], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(dmon[i, :, 4], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()
