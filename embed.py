import os

import networkx as nx
import torch
from transformers import BertTokenizer, BertModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
directory = "./dwug_en/graphs/opt"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device)
model.eval()
for file in os.listdir(directory):
    graph = nx.read_gpickle(f"{directory}/{file}")
    for node in list(graph.nodes):
        context = graph.nodes[node]['context_lemmatized']
        marked_text = "[CLS] " + context + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        index = int(graph.nodes[node]['indexes_target_token_tokenized'])
        index_word = context.split(' ')[index]
        tokenized_index = 0
        while tokenized_index < len(tokenized_text):
            if tokenized_text[tokenized_index] != index_word:
                tokenized_index += 1
            else:
                break
        if tokenized_index == len(tokenized_text):
            print(f"error {index_word}")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec.to('cpu').tolist())
        graph.nodes[node]['bert_index'] = tokenized_index
        graph.nodes[node]['bert_embedding'] = token_vecs_cat
    nx.write_gpickle(graph, f"./dwug_en/graphs/bert/raw/{file}")
