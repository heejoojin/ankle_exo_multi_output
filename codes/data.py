import torch
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
from metric import *
from torch.utils.data import Dataset

def multitask_dataset(data_list, npy_data_path, model, window_size, normalization, _mode):
    # constructing data for multi-output (regression & classification) model

    columns = ['accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'ankle_angle', 'ankle_velocity',
            'gait_phase',
            'fsr_heel_strike', 'fsr_toe_off',
            'fsr_is_stance']

    x = np.empty((0, window_size, 8))
    ry = np.empty((0, 2))
    cy = np.empty((0, 1))
    y = np.empty((0, 3))
    
    if _mode == 'test':
        heel_strike_toe_off = np.empty((0, 2))

    for file_path in data_list:

        data = pd.read_csv(file_path, usecols=columns).dropna().reset_index(drop=True).to_numpy() # fillna(0).to_numpy()
        _x = data[:, :-4]

        # normalization
        if normalization:
            _x = (_x - np.mean(_x, axis=None)) / np.std(_x, axis=None)

        _ry = data[:, -4]
        _cy = data[:, -1]
        
        # window sliding
        shape = (_x.shape[0] - window_size + 1, window_size, _x.shape[1])
        strides = (_x.strides[0], _x.strides[0], _x.strides[1])
        _x = np.lib.stride_tricks.as_strided(_x, shape=shape, strides=strides)
        
        _ry = _ry[window_size - 1:]
        _ry = gait_phase_to_polar_coordinates(_ry)

        _cy = _cy[window_size - 1:]
        _cy = np.expand_dims(_cy, axis=1)

        x = np.concatenate([x, _x], axis=0)
        ry = np.concatenate([ry, _ry], axis=0)
        cy = np.concatenate([cy, _cy], axis=0)

        if _mode == 'test':
            _heel_strike_toe_off = data[:, -3:-1]
            _heel_strike_toe_off = _heel_strike_toe_off[window_size - 1:, :]
            heel_strike_toe_off = np.concatenate([heel_strike_toe_off, _heel_strike_toe_off], axis=0)
    
    y = np.concatenate([ry, cy], axis=1)

    if model == 'cnn' or model =='tcn':
        x = np.swapaxes(x, 1, 2)
        print('convolutional network input data shape is (total batch, columns, window size)', x.shape)

    np.save(os.path.join(npy_data_path, 'x_%s.npy'%_mode), x)
    np.save(os.path.join(npy_data_path, 'y_%s.npy'%_mode), y)
    
    if _mode == 'test':
        np.save(os.path.join(npy_data_path, 'heel_strike_toe_off.npy'), heel_strike_toe_off)

def regression_dataset(data_list, npy_data_path, model, window_size, normalization, _mode):
    # constructing data for regression model

    x = np.empty((0, window_size, 8))
    y = np.empty((0, 2))
    
    columns = ['accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z', 
                'ankle_angle', 'ankle_velocity',
                'fsr_heel_strike', 'fsr_toe_off',
                'fsr_gait_phase']

    for file_path in data_list:
        data = pd.read_csv(file_path, usecols=columns).fillna(0)
        _x = data.iloc[:, :-3].to_numpy()

        # normalize input data if 'normalization' variable is set to True
        if normalization:
            _x = (_x - np.mean(_x, axis=None)) / np.std(_x, axis=None)

        _y = data.iloc[:, -1].to_numpy()
        heel_strike = data.iloc[:, -3].to_numpy()
        toe_off = data.iloc[:, -2].to_numpy()

        heel_strike_idx = np.where(heel_strike == 1)[0]
        toe_off_idx = np.where(toe_off == 1)[0]

        if heel_strike_idx.shape[0] != toe_off_idx.shape[0]:
            raise Exception('# of heel strikes is not equal to # of toe offs!')
        
        for j in range(heel_strike_idx.shape[0]):
            start_idx = int(heel_strike_idx[j] - window_size + 1)
            if start_idx < 0:
                continue
            end_idx = int(toe_off_idx[j])
            new_x = _x[start_idx:end_idx, :]
            new_y = _y[start_idx:end_idx]
            
            # window sliding
            shape = (new_x.shape[0] - window_size + 1, window_size, new_x.shape[1])
            strides = (new_x.strides[0], new_x.strides[0], new_x.strides[1])
            new_x = np.lib.stride_tricks.as_strided(new_x, shape=shape, strides=strides)
            new_y = new_y[window_size - 1:]
            new_y = gait_phase_to_polar_coordinates(new_y)
            x = np.concatenate([x, new_x], axis=0)
            y = np.concatenate([y, new_y], axis=0)
        
    if model == 'cnn' or model == 'tcn':
        x = np.swapaxes(x, 1, 2)
        print('convolutional network input data shape is (total batch, columns, window size)', x.shape)
    
    np.save(os.path.join(npy_data_path, 'x_%s.npy'%_mode), x)
    np.save(os.path.join(npy_data_path, 'y_%s.npy'%_mode), y)

def classification_dataset(data_list, npy_data_path, model, window_size, normalization, _mode):
    # constructing data for classificaiton model

    columns = ['accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'ankle_angle', 'ankle_velocity',
            'fsr_heel_strike', 'fsr_toe_off',
            'fsr_is_stance']
    x = np.empty((0, window_size, 8))
    y = np.empty((0, 1))
    
    if _mode == 'test':
        heel_strike_toe_off = np.empty((0, 2))

    for file_path in data_list:

        data = pd.read_csv(file_path, usecols=columns).fillna(0).to_numpy()
        
        _x = data[:, :-3]

        # normalize input data if 'normalization' variable is set to True
        if normalization:
            _x = (_x - np.mean(_x, axis=None)) / np.std(_x, axis=None)

        _y = data[:, -1]

        # window sliding
        shape = (_x.shape[0] - window_size + 1, window_size, _x.shape[1])
        strides = (_x.strides[0], _x.strides[0], _x.strides[1])
        _x = np.lib.stride_tricks.as_strided(_x, shape=shape, strides=strides)
        _y = _y[window_size - 1:]
        _y = np.expand_dims(_y, axis=1)
        x = np.concatenate([x, _x], axis=0)
        y = np.concatenate([y, _y], axis=0)

        if _mode == 'test':
            _heel_strike_toe_off = data[:, -3:-1]
            _heel_strike_toe_off = _heel_strike_toe_off[window_size - 1:, :]
            heel_strike_toe_off = np.concatenate([heel_strike_toe_off, _heel_strike_toe_off], axis=0)
            
    if model == 'cnn' or model =='tcn':
        x = np.swapaxes(x, 1, 2)
        print('convolutional network input data shape is (total batch, columns, window size)', x.shape)

    np.save(os.path.join(npy_data_path, 'x_%s.npy'%_mode), x)
    np.save(os.path.join(npy_data_path, 'y_%s.npy'%_mode), y)

    if _mode == 'test':
        np.save(os.path.join(npy_data_path, 'heel_strike_toe_off.npy'), heel_strike_toe_off)
    
class DataSet(Dataset):
    def __init__(self, npy_data_path, _mode):

        self.x = torch.Tensor(np.load(os.path.join(npy_data_path, 'x_%s.npy'%_mode)))
        self.y = torch.Tensor(np.load(os.path.join(npy_data_path, 'y_%s.npy'%_mode)))

    def __getitem__(self, idx):
        return self.x[idx,:,:], self.y[idx,:]

    def __len__(self):
        return self.x.size(0)
