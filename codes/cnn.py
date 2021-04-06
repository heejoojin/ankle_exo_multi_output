import os
import torch 
import torch.nn as nn
import activation, miscellaneous
    
class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        print(kwargs)
        # in_channels = kwargs['in_channels']
        # out_channels = kwargs['out_channels']
        channels = kwargs['channels']
        self.out_features = kwargs['out_features']

        kernel_size = kwargs['kernel_size']
        window_size = kwargs['window_size']
        _activation = kwargs['activation']
        dropout = kwargs['dropout']
        
        # saving model parameters
        torch.save(kwargs, os.path.join(kwargs['result_path'], 'model_params.tar'))

        # first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.act1 = activation.get_activation(_activation)
        self.drop1 = nn.Dropout(p=dropout)

        # second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(num_features=channels)
        self.act2 = activation.get_activation(_activation)
        self.drop2 = nn.Dropout(p=dropout)

        # getting output size after two convolutions
        out = miscellaneous.get_sequence_out(window_size, kernel_size)
        out = miscellaneous.get_sequence_out(out, kernel_size)

        # getting output size after "flattening"
        in_features = int(out * channels)
    
        # two fully connected layers for single-output model (regression | classification)
        self.fc1 = nn.Linear(in_features=in_features, out_features=in_features//2)
        self.fc2 = nn.Linear(in_features=in_features//2, out_features=self.out_features)
        
        # two fully connected layers for gait phase regression of a multi-output model
        self.rfc1 = nn.Linear(in_features=in_features, out_features=in_features//2)
        self.cfc1 = nn.Linear(in_features=in_features, out_features=in_features//2)

        # two fully connected layers for gait phase classification of a multi-output model
        self.rfc2 = nn.Linear(in_features=in_features//2, out_features=2)
        self.cfc2 = nn.Linear(in_features=in_features//2, out_features=1)
       
        self.init_weights()
        self.cnn = nn.Sequential(self.conv1, self.bn1, self.act1, self.drop1, self.conv2, self.bn2, self.act2, self.drop2)

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.rfc1.weight)
        torch.nn.init.kaiming_normal_(self.cfc1.weight)
        torch.nn.init.kaiming_normal_(self.rfc2.weight)
        torch.nn.init.kaiming_normal_(self.cfc2.weight)

    def forward(self, x):
        out = self.cnn(x)
        out = out.flatten(start_dim=1)

        if self.out_features != 3:
            # single-ouput model (regression | classification)
            out = self.fc1(out)
            out = self.fc2(out)
        else:
            # multi-ouput model (regression & classification)
            
            out_r = self.rfc1(out)            
            out_c = self.cfc1(out)

            out_r = self.rfc2(out_r)
            out_c = self.cfc2(out_c)

            out = (out_r, out_c)
            
        return out