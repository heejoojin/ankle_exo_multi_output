import torch
import math
import numpy as np

def classification_metrics(output: torch.Tensor, target: torch.Tensor):
    # getting accuracy, precision, recall, specificity, negative predictive value, f1score for classificaion evaluation
    # (swing = 0 | stance = 1)

    metrics = dict()
    eps = 1e-8

    tn, fn, fp, tp = confusion_matrix(output, target)
    metrics['accuracy'] = (tp + tn) / (tp + fn + fp + tn + eps)
    metrics['precision'] = tp / (tp + fp + eps)
    metrics['recall'] = tp / (tp + fn + eps)
    metrics['specificity'] = tn / (tn + fp + eps)
    metrics['npv'] = tn / (tn + fn + eps)
    metrics['f1score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision']+ metrics['recall'] + eps)

    return metrics

def confusion_matrix(output: torch.Tensor, target: torch.Tensor):
    # confusion matrix for binary classification (swing = 0 | stance = 1)

    tn, fn, fp, tp = 0, 0, 0, 0
    for _output, _target in zip(output, target):
        if _target == 0:
            if _output == 0:
                tn = tn + 1
            else:
                fn = fn + 1
        elif _target == 1:
            if _output == 0:
                fp = fp + 1
            else:
                tp = tp + 1
    return tn, fn, fp, tp

def gait_phase_to_polar_coordinates(gait_phase: np.ndarray):
    # converting gait phase [%] to x, y polar coordinates for training the model

    theta = gait_phase * 2 * np.pi
    gait_phase_x = np.cos(theta)
    gait_phase_x = np.expand_dims(gait_phase_x, axis=1)
    gait_phase_y = np.sin(theta)
    gait_phase_y = np.expand_dims(gait_phase_y, axis=1)
    polar_coordinates = np.concatenate([gait_phase_x, gait_phase_y], axis=1)
   
    return polar_coordinates

def polar_coordiantes_to_gait_percentage(output: torch.Tensor, target: torch.Tensor):
    # converting x, y polar coordinates to gait phase [%]

    output_gp = torch.remainder(torch.atan2(output[:, 1], output[:, 0]) + 2 * np.pi, 2 * np.pi) / (2 * math.pi)
    output_gp = output_gp.unsqueeze(dim=1)
    target_gp = torch.remainder(torch.atan2(target[:, 1], target[:, 0]) + 2 * np.pi, 2 * np.pi) / (2 * math.pi)
    target_gp = target_gp.unsqueeze(dim=1)

    gait_percentage = torch.cat((target_gp, output_gp), dim=1)

    return gait_percentage

def gait_phase_rmse(output: torch.Tensor, target: torch.Tensor): #, mask=None):
    
    # if mask != None:
    #     indices = torch.nonzero(mask, as_tuple=True)
    #     indices = indices[0]
    #     output = output[indices, :]
    #     target = target[indices, :]
    
    # calculating cosine distance
    num = torch.sum(output * target, dim=1) # element-wise multiplication
    denom = torch.norm(output, dim=1) * torch.norm(target, dim=1)
    cos = num / denom
    cos = torch.clamp(cos, min=-1.0, max=1.0)
    theta = torch.acos(cos) # computing the inverse cosine of each element in input
    gp_error = theta * 100 / (2 * math.pi)
    rmse = torch.sqrt(torch.mean(torch.square(gp_error)))
    # std = torch.std(gp_error)

    # rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    # std = torch.where(torch.isnan(std), torch.zeros_like(std), std)
    
    return rmse # , std