import torch
import math
import torch.nn as nn
from d2l import torch as d2l

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def prediction(prefix, num_preds, net, vocab, device):
    # initialize state
    state = net.begin_state(batch_size=1, device=device)
    # vocab[prefix[0]]: first token's index in prefix
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # warmup: just append prefix without output
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # predition for num_preds steps(append index in vocab)
        y, state = net(get_input(), state)
        # y: (seq_len×batch_size, vocab_size)
        # int(y.argmax(dim=1).reshape(1)): (seq_len×batch_size) just one dimension
        # values of element of y is predictions' indices so it can be straightly compared to truth
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train_epoch(net, train_iter, loss, updater, device, random_iter):
    state = None
    metric = [0.0] * 2  # train_loss, number of tokens
    for X, Y in train_iter:
        if state is None or random_iter:
            # initialize state
            # if random_iter, update begin_state before each batch
            # (no hidden_state information between each batch)
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # if not initial state, prevent gradient backward to h(t-1)
            # we just need to calculate gradient through h(t)
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
            
        # y_hat: 
        y_hat, state = net(X, state)
        # calculate perplexity
        # y_hat and y.long() 
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metrics_list = [l * y.numel(), y.numel()]
        metric = [a + float(b) for a, b in zip(metric, metrics_list)]
    return math.exp(metric[0] / metric[1])


def train_model(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    print('Training on', device)
    if isinstance(net, nn.Module):
        net.to(device)
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: prediction(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0: # prediction every 10 epochs
            print(predict('time traveller'))
    print(f'Perplexity: {ppl:.1f}, {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    
    