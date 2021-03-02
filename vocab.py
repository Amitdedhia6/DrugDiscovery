from common import base_data_path, max_sequence_length
import os
import math
from typing import List, Dict
import torch
from random import randint


class Vocab():
    def __init__(self):
        self.vocab_size: int = 0
        self.character_embedding_size: int = 2
        self._embeddings_tensor: torch.FloatTensor = None
        self._element_to_index: Dict[str, int] = {}
        self._index_to_element: Dict[int, str] = {}
        self._init_vocab()

    def _init_vocab(self):
        filepath = os.path.join(base_data_path, "vocab.txt")
        f = open(filepath, "r")
        elements_list = f.read().splitlines()
        elements_list.append(' ')
        f.close()

        total_elements = len(elements_list)
        angle_increment = 360 / total_elements
        current_angle = 0
        index = 0
        embedding_list = []

        for element in elements_list:
            embedding_list.append([math.cos(current_angle), math.sin(current_angle)])
            self._element_to_index[element] = index
            self._index_to_element[index] = element
            current_angle += angle_increment
            index += 1

        self.vocab_size = len(self._element_to_index)
        self._embeddings_tensor = torch.FloatTensor(embedding_list)

    def get_embedding(self, element: str) -> torch.FloatTensor:
        if element in self._element_to_index:
            index = self._element_to_index[element]
            return self._embeddings_tensor[index]
        else:
            raise ValueError('Inappropriate argument to function get_embedding of Vocab class')

    def get_element_from_index(self, index: int) -> str:
        if index in self._index_to_element:
            return self._index_to_element[index]
        else:
            raise ValueError('Inappropriate argument to function get_element_from_index of Vocab class')

    def get_index_from_element(self, element: str) -> int:
        if element in self._element_to_index:
            return self._element_to_index[element]
        else:
            raise ValueError('Inappropriate argument to function get_index_from_element of Vocab class')

    def split_smiles_repr(self, smile_repr: str) -> List[str]:
        element_list = []
        skip_next = False
        for i in range(len(smile_repr)):
            if skip_next:
                skip_next = False
                continue

            element = smile_repr[i]
            if (i < (len(smile_repr) - 1)) and (smile_repr[i].isalpha()):
                possible_element = element + smile_repr[i+1]
                if possible_element in self._element_to_index:
                    element = possible_element
                    skip_next = True

            if element in self._element_to_index:
                element_list.append(element)
            else:
                raise ValueError('Inappropriate argument to function get_elements_from_smiles_data of Vocab class')
        return element_list

    def _get_single_random_element(self):
        value = randint(0, len(self._index_to_element) - 1)
        c = self._index_to_element[value]
        if c.isalpha():
            return c
        else:
            return self._get_single_random_element()

    def get_random_elements(self, batch_size):
        element_list = []

        for _i in range(batch_size):
            element_list.append(self._get_single_random_element())

        return element_list

    # def get_element_from_embedding(self, embedding: torch.Tensor) -> str:
    #    distance = 100000000.0
    #    element = ""
    #    for elem in self._element_to_index:
    #        idx = self._element_to_index[elem]
    #        elem_embedding = np.array(self._embeddings_tensor[idx])
    #        d = np.linalg.norm(elem_embedding - embedding.detach().numpy() )
    #        if d < distance:
    #            distance = d
    #            element = elem
    #    return element

    def get_sequences_from_embeddings(self, embeddings: torch.Tensor, e_len: torch.Tensor) -> str:
        with torch.no_grad():
            input_embeddings = embeddings.unsqueeze(2)
            existing_embeddings = self._embeddings_tensor.unsqueeze(0).unsqueeze(0)
            diff = input_embeddings - existing_embeddings
            diff_0 = torch.index_select(diff, 3, torch.LongTensor([0]))
            diff_1 = torch.index_select(diff, 3, torch.LongTensor([1]))

            diff_0 = diff_0 ** 2
            diff_1 = diff_1 ** 2

            final = (diff_0 + diff_1).sqrt().squeeze(-1)
            element_indices = final.argmin(2)

            sequences = [' '] * len(embeddings)
            eos_index = self.get_index_from_element(' ')
            index = 0
            for s in element_indices:
                s_ = []
                for e in s:
                    if e.item() == eos_index:
                        break
                    s_.append(self._index_to_element[e.item()])
                length = len(s_)

                sequences[index] = "".join(s_)
                if (e_len[index] != length):
                    print(". . . Assert breaking now")
                    print(". . . ", e_len[index], length)
                    print(". . . ", sequences[index])
                    print(". . . ", s)
                assert(e_len[index] == length)
                index += 1

            return sequences

    def is_end_of_sequence(self, embeddings: torch.FloatTensor) -> torch.BoolTensor:
        #   d = torch.dist(embedding, elem_embedding).item()
        #   d = math.sqrt(math.pow(embedding[0] - elem_embedding[0], 2) + math.pow(embedding[1] - elem_embedding[1], 2))
        #   d = np.linalg.norm((embedding - elem_embedding).detach().cpu().numpy())

        with torch.no_grad():
            input_embeddings = embeddings.reshape(-1, 1, self.character_embedding_size)
            existing_embeddings = self._embeddings_tensor.reshape(1, -1, self.character_embedding_size)
            diff = input_embeddings - existing_embeddings
            diff_0 = torch.index_select(diff, 2, torch.LongTensor([0]))
            diff_1 = torch.index_select(diff, 2, torch.LongTensor([1]))

            diff_0 = diff_0 ** 2
            diff_1 = diff_1 ** 2

            final = (diff_0 + diff_1).sqrt().squeeze(-1)
            element_indices = final.argmin(1)
            eos_index = self.get_index_from_element(' ')
            final_result = (element_indices == eos_index)
            return final_result

    def get_tensors_from_element_list(self, element_data: List[List[str]]):
        vector_size = self.character_embedding_size
        result = torch.zeros((len(element_data), max_sequence_length, vector_size), dtype=torch.float32)

        for i, element_list in enumerate(element_data):
            if len(element_list) > max_sequence_length:
                continue

            for j, element in enumerate(element_list):
                e = self.get_embedding(element)
                result[i, j] = e

        return result
