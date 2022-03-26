import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle
import os
import pandas as pd
with open(r"MinCut21.pickle", "rb") as output_file:
    mincut21 = np.array(pickle.load(output_file))
with open(r"MinCut20.pickle", "rb") as output_file:
    mincut20 = np.array(pickle.load(output_file))
with open(r"MinCut11.pickle", "rb") as output_file:
    mincut11 = np.array(pickle.load(output_file))
with open(r"MinCut10.pickle", "rb") as output_file:
    mincut10 = np.array(pickle.load(output_file))
with open(r"DMoN21.pickle", "rb") as output_file:
    dmon21 = np.array(pickle.load(output_file))
with open(r"DMoN20.pickle", "rb") as output_file:
    dmon20 = np.array(pickle.load(output_file))
with open(r"DMoN11.pickle", "rb") as output_file:
    dmon11 = np.array(pickle.load(output_file))
with open(r"DMoN10.pickle", "rb") as output_file:
    dmon10 = np.array(pickle.load(output_file))
with open(r"kmean0.pickle", "rb") as output_file:
    kmean0 = np.array(pickle.load(output_file))
with open(r"kmean1.pickle", "rb") as output_file:
    kmean1 = np.array(pickle.load(output_file))


directory = "./dwug_en/graphs/bert_balance/raw"
words1 = [file.split('_')[0] for file in os.listdir(directory)]
words0 = [file.split('_')[0] for file in os.listdir("./dwug_en/graphs/bert/raw")]
word_dict = {}

for word in words1:
    word_dict[word] = np.array([mincut10[words0.index(word), -1, 3:], 
                                mincut11[words1.index(word), -1, 3:],
                                mincut20[words0.index(word), -1, 3:],
                                mincut21[words1.index(word), -1, 3:],
                                dmon10[words0.index(word), -1, 3:], 
                                dmon11[words1.index(word), -1, 3:],
                                dmon20[words0.index(word), -1, 3:],
                                dmon21[words1.index(word), -1, 3:],
                                kmean0[words0.index(word)][1:],
                                kmean1[words1.index(word)][1:]
                            ])
table = pd.DataFrame(dict(word=word_dict.keys(), 
                          m10N=[float(word_dict[word][0,0]) for word in word_dict.keys()],
                          m10r=[float(word_dict[word][0,-1]) for word in word_dict.keys()],
                          m20N=[float(word_dict[word][2,0]) for word in word_dict.keys()],
                          m20R=[float(word_dict[word][2,-1]) for word in word_dict.keys()],
                          d10N=[float(word_dict[word][4,0]) for word in word_dict.keys()],
                          d10R=[float(word_dict[word][4,-1]) for word in word_dict.keys()],
                          d20N=[float(word_dict[word][6,0]) for word in word_dict.keys()],
                          d20R=[float(word_dict[word][6,-1]) for word in word_dict.keys()],
                          k0N=[float(word_dict[word][8,0]) for word in word_dict.keys()],
                          k0R=[float(word_dict[word][8,-1]) for word in word_dict.keys()],
                          
                        )
                    )

table1 = pd.DataFrame(dict(word=word_dict.keys(), 
                          m11N=[float(word_dict[word][1,0]) for word in word_dict.keys()],
                          m11r=[float(word_dict[word][1,-1]) for word in word_dict.keys()],
                          m21N=[float(word_dict[word][3,0]) for word in word_dict.keys()],
                          m21R=[float(word_dict[word][3,-1]) for word in word_dict.keys()],
                          d11N=[float(word_dict[word][5,0]) for word in word_dict.keys()],
                          d11R=[float(word_dict[word][5,-1]) for word in word_dict.keys()],
                          d21N=[float(word_dict[word][7,0]) for word in word_dict.keys()],
                          d21R=[float(word_dict[word][7,-1]) for word in word_dict.keys()],
                          k1N=[float(word_dict[word][9,0]) for word in word_dict.keys()],
                          k1R=[float(word_dict[word][9,-1]) for word in word_dict.keys()],
                          
                        )
                    )

                    
print(table.to_latex(index=False, float_format="%.2f")) 
    
print(table.to_latex(index=False, float_format="%.2f")) 





def plot_together(index, name, threshold):
    select_index = np.argwhere(dmon21[:, -1, 3] > threshold).flatten()
    select_word = [words1[int(i)] for i in select_index]
    select_index0 = [words0.index(word) for word in select_word]

    dmon21 = dmon21[select_index, :, :]
    dmon20 = dmon20[select_index, :, :]
    mincut20 = mincut20[select_index, :, :]
    mincut21 = mincut21[select_index, :, :]
    color = ['r', 'g', 'b', 'c', 'm']
    [plt.plot(mincut20[i, :, index],  color=color[i]) for i in range(len(select_word))]
    [plt.plot(mincut21[i, :, index],  color=color[i], linestyle=":") for i in range(len(select_word))]
    [plt.plot(dmon20[i, :, index],  color=color[i], linestyle="--") for i in range(len(select_word))]
    [plt.plot(dmon21[i, :, index],  color=color[i], linestyle="-.") for i in range(len(select_word))]
    custom_lines = [Line2D([0], [0], ls="-"),
                    Line2D([0], [0], ls="--"),
                    Line2D([0], [0], ls=":"),
                    Line2D([0], [0], ls="-.")] + [Line2D([0], [0], c=c) for c in color]
    plt.legend(custom_lines, ['MinCut 1 layer', 'DMoN 1 layer', 'MinCut 2 layer', 'DMoN 2 layer']+select_word)
    plt.xlabel("epochs")
    plt.ylabel(name)
    plt.show()


#plot_together(3, "NMI", 0.9)
