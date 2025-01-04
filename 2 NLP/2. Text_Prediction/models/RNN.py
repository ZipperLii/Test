import torch
import torch.nn as nn
from torch.nn import functional as F


class RNNModel(nn.Module):
    def __init__(self,  vocab_size, num_hiddens, gate_type=None, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        if gate_type == 'GRU':
            self.rnn = nn.GRU(vocab_size, num_hiddens)
        elif gate_type == 'LSTM':
            self.rnn = nn.LSTM(vocab_size, num_hiddens)
        else:
            self.rnn = nn.RNN(vocab_size, num_hiddens)
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # if RNN is not bidirectional, num_directions = 1
        # if RNN is bidirectional, num_directions = 2
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # add a fully connected layer to output
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            # because the output of a bidirectional RNN is the concatenation of the outputs of two RNNs
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # the input shape of nn.RNN is (seq_len, batch, input_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # this Y is the output of the last hidden layer
        Y, state = self.rnn(X, state)
        # The fully connected layer first reshapes Y to (time steps * batch size, number of hidden units)
        # Its output shape is (time steps * batch size, vocab size).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
        