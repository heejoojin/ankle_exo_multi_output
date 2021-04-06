import torch
import torch.nn as nn

def get_criterion(criterion):

    if criterion == 'mae':
        # mean absolute error
        return nn.L1Loss()
    elif criterion == 'mse':
        return nn.MSELoss()
    elif criterion == 'rmse':
        # root mean squre error
        return RMSELoss()
    elif criterion == 'cross_entropy':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target))
        
