import torch
import torch.nn as nn
import os
from common import base_data_path
from typing import List
import pandas as pd


CONTEXT_SIZE = 1  # 1 words to the left, 1 to the right
EMDEDDING_DIM = 3
word_to_ix = {}
ix_to_word = {}


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]


def split_smiles_repr(smile_repr: str) -> List[str]:
    element_list = []
    skip_next = False
    for i in range(len(smile_repr)):
        if skip_next:
            skip_next = False
            continue

        element = smile_repr[i]
        if (i < (len(smile_repr) - 1)) and (smile_repr[i].isalpha()):
            possible_element = element + smile_repr[i+1]
            if possible_element in word_to_ix:
                element = possible_element
                skip_next = True

        if element in word_to_ix:
            element_list.append(element)
        else:
            raise ValueError('Inappropriate argument to function get_elements_from_smiles_data of Vocab class')
    return element_list


def get_data(sequence_list: List[str]):
    _sequence_list = []
    sequence_elements_list = []

    for s in sequence_list:
        split_elements = split_smiles_repr(s)
        _sequence_list.append(s)
        sequence_elements_list.append(split_elements)

    return sequence_elements_list


filepath = os.path.join(base_data_path, "vocab.txt")
f = open(filepath, "r")
elements_list = f.read().splitlines()
elements_list.append(' ')
f.close()

vocab = elements_list
vocab_size = len(elements_list)

for i, word in enumerate(vocab):
    word_to_ix[word] = i
    ix_to_word[i] = word

filepath = os.path.join(base_data_path, "dataset_v1.csv")
df = pd.read_csv(filepath, sep=",", header=0)
smiles_data = get_data(df.SMILES.tolist())


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.LongTensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)


model = CBOW(vocab_size, EMDEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for epoch in range(50):
    total_loss = 0
    for smiles_element_list in smiles_data:
        for i in range(1, len(smiles_element_list) - 1):
            context = [smiles_element_list[i - 1], smiles_element_list[i + 1]]
            target = smiles_element_list[i]
            context_vector = make_context_vector(context, word_to_ix)
            model.zero_grad()
            log_probs = model(context_vector)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    print(f"Epoch - {epoch}, Loss - {total_loss}")
