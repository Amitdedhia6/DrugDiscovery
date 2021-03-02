import numpy as np
from common import max_sequence_length
import torch
from typing import List, Dict


class SmilesData():
    def __init__(self, vocab):
        self.vocab = vocab
        self._sequence_tensors: Dict[str, ] = {}

    def fill(self, sequence_list: List[str]):
        _sequence_list = []
        sequence_elements_list = []
        sequence_length_list = []

        for s in sequence_list:
            split_elements = self.vocab.split_smiles_repr(s)
            if len(split_elements) >= max_sequence_length:
                continue
            else:
                split_elements.append(' ')
                _sequence_list.append(s)
                sequence_elements_list.append(split_elements)
                sequence_length_list.append(len(split_elements))

        self.sequence_list = _sequence_list
        self.sequence_list_size = len(_sequence_list)
        self.sequence_tensors = self.vocab.get_tensors_from_element_list(sequence_elements_list)
        self.sequence_length_data = torch.from_numpy(np.array(sequence_length_list))
