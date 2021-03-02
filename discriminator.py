import torch
from torch import nn
from common import device
from vocab import Vocab
import torch.nn.utils.rnn as rnn_utils


class Discriminator(nn.Module):
    """
    discriminative neural network
    """
    def __init__(self, vocab: Vocab):
        super(Discriminator, self).__init__()
        self.vocab = vocab
        self.embedding_dim = self.vocab.character_embedding_size
        self.n_lstm_layers = 2
        self.hidden_dim = 5
        self.output_size = 1

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_lstm_layers, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_dim, self.output_size).to(device)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None
        self.cell_state = None

    def forward(self, x, ln):
        batch_size = x.size(0)

        self.hidden = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim).to(device)

        x_packed = rnn_utils.pack_padded_sequence(x, ln, batch_first=True, enforce_sorted=False)

        lstm_out, (self.hidden, self.cell_state) = self.lstm(x_packed, (self.hidden, self.cell_state))
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        unpacked = self._process_unpacked_sequence(unpacked, unpacked_len)
        lstm_out = unpacked
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.fc(lstm_out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out

    def _process_unpacked_sequence(self, unpacked, unpacked_len):
        tensor_array = []
        for index in range(len(unpacked)):
            t = unpacked[index, unpacked_len[index] - 1, :]
            t = torch.reshape(t, (1, -1))
            tensor_array.append(t)
        return torch.cat(tensor_array, dim=0)
