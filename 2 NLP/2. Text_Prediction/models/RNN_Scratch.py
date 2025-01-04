import torch
from torch import nn
from torch.nn import functional as F


def get_params(vocab_size, num_hiddens, gate_type, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    def normal_group():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )
    W_xh, W_hh, b_h = normal_group()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    if gate_type == 'GRU':
        W_xr, W_hr, b_r = normal_group()
        params += [W_xr, W_hr, b_r]  # add reset gate params
        W_xz, W_hz, b_z = normal_group()
        params += [W_xz, W_hz, b_z]  # add update gate params
    elif gate_type == 'LSTM':
        W_xf, W_hf, b_f = normal_group()
        params += [W_xf, W_hf, b_f]
        W_xi, W_hi, b_i = normal_group()
        params += [W_xi, W_hi, b_i]
        W_xc, W_hc, b_c = normal_group()
        params += [W_xc, W_hc, b_c]
        W_xo, W_ho, b_o = normal_group()
        params += [W_xo, W_ho, b_o]
    
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, gate_type, device):
    init_hidden = (torch.zeros((batch_size, num_hiddens), device=device), )
    if gate_type == 'LSTM':
        init_hidden = (torch.zeros((batch_size, num_hiddens), device=device),
                       torch.zeros((batch_size, num_hiddens), device=device))
    return init_hidden


def rnn_forward(inputs, state, gate_type, params):
    W_xh, W_hh, b_h, W_hq, b_q = params[:5]
    if gate_type == 'GRU':
        W_xr, W_hr, b_r = params[5:8]
        W_xz, W_hz, b_z = params[8:]
        H = state[0]
    elif gate_type == 'LSTM':
        W_xf, W_hf, b_f = params[5:8]
        W_xi, W_hi, b_i = params[8:11]
        W_xc, W_hc, b_c = params[11:14]
        W_xo, W_ho, b_o = params[14:]
        (H, Memory) = state
    else:
        H = state[0]
    outputs = []
    # inputs: (seq_len, batch_size, vocab_size)
    for X in inputs:  # X: (batch_size, vocab_size), W_xh: (vocab_size, hidden_size)
        if gate_type == 'GRU':
            R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
            Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
            candH = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
            H = Z * H + (1 - Z) * candH
            hidden_out = (H,)
        elif gate_type == 'LSTM':
            F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
            I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
            C = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
            O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
            Memory = F * Memory + I * C
            H = O * torch.tanh(Memory)
            hidden_out = (H, Memory)
        else:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            hidden_out = (H,)
        
        Y = torch.mm(H, W_hq) + b_q
        # outputs: (seq_len, batch_size, vocab_size)
        outputs.append(Y)  # Y: (batch_size, vocab_size)
    # merge seq_len and batch_size → (seq_len × batch_size, vocab_size)
    return torch.cat(outputs, dim=0), hidden_out


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, gate_type=None):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens,
                                 gate_type, device)
        self.gate_type = gate_type
        
    def __call__(self, X, state):
        # X input size: (batch_size × seq_len)
        # in each sequence，a token corresponds to an index
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # after one_hot f: (seq_len × batch_size × vocab_size)
        # vocab_size = dimension of one-hot vector
        return rnn_forward(X, state,
                           self.gate_type, self.params)
        
    def begin_state(self, batch_size, device):
        return init_rnn_state(batch_size, self.num_hiddens,
                              self.gate_type, device)
    
    