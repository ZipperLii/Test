import torch
import re
from d2l import torch as d2l
from utils.data_utils import SeqDataLoader

def data_loader(PATH, batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(PATH, 
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    # 生成初始状态
    state = net.begin_state(batch_size=1, device=device)
    # 把prefix第一个词在词表的index放入output
    outputs = [vocab[prefix[0]]]
    # 把output最近预测的那个词的tensor输入(outputs[-1]表示最后的预测)
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期将输入的话重新输入，从而使隐藏状态从初始化开始预热更新
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        # 这一步仅仅将prefix中的输入append到output中
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        # 将输出的one-hot码append到output中
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 最后将index转化为token
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def main():
    pass
    
    
if __name__ == '__main__':
    main()