import torch
from torch import nn
from common import device, max_sequence_length, noise_vector_length, noise


class Generator(torch.nn.Module):
    """
    A generative neural network
    """
    def __init__(self, vocab):
        super(Generator, self).__init__()

        self.vocab = vocab
        self.input_dim = noise_vector_length + vocab.character_embedding_size
        self.hidden_dim = 5
        self.n_lstm_layers = 1
        self.output_size = vocab.character_embedding_size
        self.hidden = None
        self.cell_state = None

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_lstm_layers, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_dim, self.output_size).to(device)
        self.tanh = nn.Tanh()

    def forward(self, batch_size):
        self.hidden = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim).to(device)

        y_list = []
        start_elements = self.vocab.get_random_elements(batch_size)
        start_elements_tensors_list = []
        for _j, element in enumerate(start_elements):
            start_elements_tensors_list.append(self.vocab.get_embedding(element))

        x = torch.stack(start_elements_tensors_list, dim=0).to(device)
        y_list.append(x)
        length_data = torch.LongTensor([1] * batch_size)
        sequence_filled = torch.BoolTensor([False] * batch_size)

        noise_singal = noise(batch_size).to(device)
        x = torch.cat((x, noise_singal), dim=1).reshape(batch_size, 1, -1)
        length_sequence_processed = 1

        while length_sequence_processed < max_sequence_length:
            lstm_out, (self.hidden, self.cell_state) = self.lstm(x, (self.hidden, self.cell_state))
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            fc_out = self.fc(lstm_out)
            a1 = torch.pow(fc_out, 2)
            a2 = torch.sum(a1, dim=1)
            a3 = torch.sqrt(a2).unsqueeze(-1)
            y = fc_out / a3
            y_list.append(y)
            length_sequence_processed += 1

            is_end_of_seq = self.vocab.is_end_of_sequence(y)
            sequence_filled = sequence_filled + is_end_of_seq
            length_increment = (sequence_filled == False).type(torch.long)
            length_data = length_data + length_increment

            noise_singal = noise(batch_size).to(device)
            x = torch.cat((y, noise_singal), dim=1).reshape(batch_size, 1, -1)

        y_final = torch.stack(y_list, dim=1).to(device)
        l_final = length_data

        return y_final, l_final

    def get_sequences_from_tensor(self, t: torch.Tensor, length: torch.Tensor):
        return self.vocab.get_sequences_from_embeddings(t, length)
