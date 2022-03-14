import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
with open(r"mincut.pickle", "rb") as output_file:
    mincut = np.array(pickle.load(output_file))
with open(r"dmon.pickle", "rb") as output_file:
    dmon = np.array(pickle.load(output_file))

directory = "./dwug_en/graphs/bert/raw"
words = [file.split('_')[0] for file in os.listdir(directory)]
[plt.plot(mincut[i, :, 2], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(mincut[i, :, 3], label=[words[i]]) for i in range(len(words))]
plt.legend()
plt.show()

[plt.plot(mincut[i, :, 4], label=[words[i]]) for i in range(len(words))]
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
