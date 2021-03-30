import os
import random
import shutil
import math
import config as c

def get_sequence_out(sequence_in, kernel_size, padding=0, dilation=1, stride=1):
    sequence_out = (sequence_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(sequence_out)

def organize_data():
    
    if os.path.isdir(c.DATA_PATH) and c.DATA_PATH != c.ORIGINAL_DATA_PATH:
        shutil.rmtree(c.DATA_PATH)
    os.mkdir(c.DATA_PATH)

    for data_type in ['all', 'left', 'right']:
        dst_dir = os.path.join(c.DATA_PATH, data_type)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

    for file_name in os.listdir(c.ORIGINAL_DATA_PATH):
        src = os.path.join(c.ORIGINAL_DATA_PATH, file_name)
        if 'LEFT' in file_name:
            dst_dir = os.path.join(os.path.join(c.DATA_PATH, 'left'), file_name)

        elif 'RIGHT' in file_name:
            dst_dir = os.path.join(os.path.join(c.DATA_PATH, 'right'), file_name)
        else:
            raise Exception('Weird data found: %s!'%file_name)
        
        shutil.copyfile(src, dst_dir)
        shutil.copyfile(src, os.path.join(os.path.join(c.DATA_PATH, 'all'), file_name))

def get_data_list(args, data_path):
    train_val_list = []
    test_list = []

    if args.test_type != 'all':
        for file_name in os.listdir(data_path):
            
            if '.csv' in file_name:
                if args.test_type in file_name:
                    test_list.append(os.path.join(data_path, file_name))
                else:
                    train_val_list.append(os.path.join(data_path, file_name))
    else:
        data_list = []
        
        for file_name in os.listdir(data_path):
            if '.csv' in file_name:
                data_list.append(file_name)

        random.Random(args.seed).shuffle(data_list)
        idx = max(math.floor(len(data_list) * args.test_ratio), 1)

        _data_list = [os.path.join(data_path, file_name) for file_name in data_list]
        train_val_list = _data_list[:-idx]
        test_list = _data_list[-idx:]
        
    return train_val_list, test_list

def split_train_val(data_list, k_fold, fold_idx):
    size = max(math.floor(len(data_list) / k_fold), 1)
    train_list = data_list[:size * (fold_idx - 1)] + data_list[size * fold_idx:]
    val_list = data_list[size * (fold_idx - 1) : size * fold_idx]

    return train_list, val_list

def split_speed_act():
    for test_type in c.TRAIN_TYPE_LIST:
        data_path = os.path.join(c.DATA_PATH, test_type)
        
        for file_name in os.listdir(data_path):
            
            speed_path = os.path.join(data_path, 'speed' + file_name.split('_')[1][-1])
            act_path = os.path.join(data_path, 'act' + file_name.split('_')[2][-1])

            for type_path in [speed_path, act_path]:
                if not os.path.isdir(type_path):
                    os.mkdir(type_path)
                
            src = os.path.join(data_path, file_name)
            shutil.copyfile(src, os.path.join(speed_path, file_name))
            shutil.copyfile(src, os.path.join(act_path, file_name))

def get_result_dirs(args):
    result_path = os.path.join(c.RESULT_PATH, args.save_name)
    npy_data_path = os.path.join(result_path, 'npy_data')
    for path in [c.RESULT_PATH, c.PLOT_PATH, result_path, result_path, npy_data_path]:
        if not os.path.isdir(path):
            os.mkdir(path)
    return result_path, npy_data_path