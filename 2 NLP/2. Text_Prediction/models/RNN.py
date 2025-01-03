import torch
from torch import nn
from torch.nn import functional as F


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn_forward(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # inputs: (seq_len, batch_size, vocab_size)
    for X in inputs:  # X: (batch_size, vocab_size), W_xh: (vocab_size, hidden_size)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        # outputs: (seq_len, batch_size, vocab_size)
        outputs.append(Y)  # Y: (batch_size, vocab_size)
    
    # merge seq_len and batch_size → (seq_len × batch_size, vocab_size)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        
    def __call__(self, X, state):
        # X input size: (batch_size × seq_len)
        # in each sequence，a token corresponds to an index
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # after one_hot f: (seq_len × batch_size × vocab_size)
        # vocab_size = dimension of one-hot vector
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
    