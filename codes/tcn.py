import os
import torch 
import torch.nn as nn
from torch.nn.utils import weight_norm
from miscellaneous import *

# resources:
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://github.com/pytorch/pytorch/issues/1333

class CausalConv1d(nn.Module):
    def __init__(self, padding):
        super(CausalConv1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        # adding padding only on the left-side
        return x[:, :, :-self.padding].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride, padding, dilation, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.causual1 = CausalConv1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.causual2 = CausalConv1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.res = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.res = None

        self.temporal_block = nn.Sequential(self.conv1, self.causual1, self.relu1, self.dropout1,
                                 self.conv2, self.causual2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.res:
            self.res.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.temporal_block(x)
        if self.res:
            _x = self.res(x)
        else:
            _x = x.clone()
        return self.relu(out + _x)

class TCN(nn.Module):
    def __init__(self, **kwargs):
        super(TCN, self).__init__()

        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        out_features = kwargs['out_features']
        kernel_size = kwargs['kernel_size']
        dropout = kwargs['dropout']
        window_size = kwargs['window_size']

        torch.save(kwargs, os.path.join(kwargs['result_path'], 'model_params.tar'))

        layers = []
        K = 3
        for k in range(K):
            dilation = 2 ** k
            temporal_block = TemporalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=1, padding=(kernel_size - 1) * dilation, 
                                            dilation=dilation, dropout=dropout)
            layers.append(temporal_block)

        self.tcn = nn.Sequential(*layers)
        in_features = int(out_channels * window_size)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.tcn(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out