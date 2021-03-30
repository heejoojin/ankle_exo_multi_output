import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'lrelu':
        return nn.LeakyReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'rlrelu':
        # https://paperswithcode.com/method/rrelu#:~:text=Randomized%20Leaky%20Rectified%20Linear%20Units%2C%20or%20RReLU%2C%20are%20an%20activation,U%20(%20l%20%2C%20u%20)%20.
        return nn.RReLU() 
    else:
        raise NotImplementedError('Unimplemented activation!')