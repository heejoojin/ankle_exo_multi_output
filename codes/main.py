import os
import argparse
import torch
import random
import config as c
from torch.utils.data import DataLoader
from test import *
from data import *
from optimizer import *
from train import *
from lstm import *
from tcn import *
from cnn import *
from criterion import *
from miscellaneous import *
from scheduler import *
from plot import *
from miscellaneous import *

def main(args):
    
    print(args.save_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not device:
        raise SystemError('There is no device!')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'train':
        main_params = args.__dict__
        print(main_params)

        data_path = os.path.join(c.DATA_PATH, args.data_type)
        result_path, npy_data_path = get_result_dirs(args)
        torch.save(main_params, os.path.join(result_path, 'main_params.tar'))

        train_val_list, test_list = get_data_list(args, data_path)
        
        for fold_idx in range(1, args.k_fold +  1):
            train_list, val_list = split_train_val(train_val_list, args.k_fold, fold_idx)

            print('train dataset', train_list, len(train_list))
            print('valdiation dataset', val_list, len(val_list))
            print('test dataset', test_list, len(test_list))

            data_list = [train_list, val_list, test_list]
            mode_list = ['train', 'val', 'test']
            for _mode, _data_list in zip(mode_list, data_list):
                if args.task == 'classification':
                    classification_dataset(data_list=_data_list, npy_data_path=npy_data_path,
                                            model=args.model, window_size=args.window_size, normalization=args.normalization, _mode=_mode)
                elif args.task == 'regression':
                    regression_dataset(data_list=_data_list, npy_data_path=npy_data_path,
                                        model=args.model, window_size=args.window_size, normalization=args.normalization, _mode=_mode)
                elif args.task == 'multi':
                    multitask_dataset(data_list=_data_list, npy_data_path=npy_data_path,
                                model=args.model, window_size=args.window_size, normalization=args.normalization, _mode=_mode)

            train_dataset = DataSet(npy_data_path=npy_data_path, _mode='train')
            val_dataset = DataSet(npy_data_path=npy_data_path, _mode='val')
            train_data_loader = DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=args.shuffle, pin_memory=True)
            val_data_loader = DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=args.shuffle, pin_memory=True)

            if args.model == 'cnn':
                model = CNN(in_channels=train_dataset[:][0].size(1), out_channels=train_dataset[:][0].size(1), out_features=train_dataset[:][1].size(-1),
                            activation=args.activation,
                            kernel_size=args.kernel_size, dropout=args.dropout,
                            window_size=args.window_size,
                            result_path=result_path).to(device)
            elif args.model == 'tcn':
                model = TCN(in_channels=train_dataset[:][0].size(1), out_channels=train_dataset[:][0].size(1), out_features=train_dataset[:][1].size(-1),
                            kernel_size=args.kernel_size, dropout=args.dropout, result_path=result_path,
                            window_size=args.window_size).to(device)
            
            optimizer = get_optimizer(args, model)
            is_plateau, scheduler = get_scheduler(args, optimizer)

            criterion = None
            if args.task == 'multi':
                criterion = {'regression': get_criterion('mse'), 'classification': get_criterion('cross_entropy')}
            else:
                criterion = get_criterion(args.criterion)

            trainer = Train(device=device, model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                            task=args.task, fold_idx=fold_idx, 
                            epochs=args.epochs, batch_size=args.batch_size, period=args.period,
                            train_data_loader=train_data_loader, val_data_loader=val_data_loader,
                            result_path=result_path, is_plateau=is_plateau)
            trainer.do()

    elif args.mode == 'test':
        
        data_path = os.path.join(c.DATA_PATH, args.data_type)
        result_path, npy_data_path = get_result_dirs(args)

        test_dataset = DataSet(npy_data_path=npy_data_path, _mode='test')
        test_data_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True)
        tester = Test(device=device, k_fold=args.k_fold, task=args.task, data_type=args.data_type, data_loader=test_data_loader, result_path=result_path)
        tester.do()
    
    elif args.mode == 'plot':

        plot_all_edges()
        plot_grouped_rmse()

    elif args.mode == 'rawdata':
        
        # plotting raw input data
        plot_raw_data()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg('--seed', type=int, default=777)

    arg('--save_name', type=str, default='')
    arg('--mode', type=str, default='train', choices=['train', 'test', 'plot', 'rawdata'])
    arg('--task', type=str, default='multi', choices=['classification', 'regression', 'multi'])
    
    arg('--data_type', type=str, default='left', choices=['left', 'right'])
    arg('--test_type', type=str, default='all', help='speed(n) | act(n) | all')

    arg('--normalization', action='store_true') # default value = False
    arg('--shuffle', action='store_false') # default value = True
    
    arg('--num_workers', type=int, default=0)
    
    arg('--k_fold', type=int, default=c.K_FOLD)

    arg('--epochs', type=int, default=1)
    arg('--window_size', type=int, default=120)
    arg('--batch_size', type=int, default=128)

    arg('--model', type=str, default='cnn', choices=['cnn', 'tcn'])
    arg('--period', type=int, default='1', help='period for saving trained models')
    
    arg('--criterion', type=str, default=None, choices=['cross_entropy', 'mse'])
    arg('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop'])
    arg('--scheduler', type=str, default='plateau', choices=['cosine', 'step', 'plateau', 'lambda'])
    arg('--activation', type=str, default='relu', choices=['relu', 'tanh', 'lrelu', 'sigmoid', 'rlrelu'])
    arg('--lr', type=float, default=1e-4)

    arg('--weight_decay', type=float, default=0.0)
    arg('--patience', type=int, default=4, help='plateau scheduler patience parameter')
    
    arg('--kernel_size', type=int, default=40, help='convolutional network hyperparameter')
    arg('--dropout', type=float, default=0.2)

    arg('--test_ratio', type=float, default=0.1)

    args = parser.parse_args()
    
    main(args)