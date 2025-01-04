from d2l import torch as d2l
from utils.data_utils import data_loader
from utils.train_utils import train_model
from models.RNN import RNNModel
from models.RNN_Scratch import RNNModelScratch
from utils.train_utils import try_gpu


def main():
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
    batch_size = 32
    num_steps = 35
    PATH = d2l.download('time_machine')
    data_iter, vocab = data_loader(PATH, batch_size, num_steps)

    hidden_size = 256
    num_epochs = 500
    device = try_gpu()
    
    model = RNNModelScratch(len(vocab), hidden_size,
                            device, 'LSTM')
    
    train_model(net=model,
                train_iter=data_iter,
                vocab=vocab,
                lr=1,
                num_epochs=num_epochs,
                device=device)
    
    
if __name__ == '__main__':
    main()